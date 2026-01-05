import re
import math
import json
import asyncio
from google import genai
from dotenv import load_dotenv
import google.genai.types as types
from typing import List, Dict, Optional


load_dotenv(".env")

def parse_json_from_text(text: str) -> Optional[Dict]:
	"""
	Extract and parse JSON from text response.
	"""
	try:
		return json.loads(text)
	except json.JSONDecodeError:
		# Try to extract JSON from text if it contains markdown code blocks
		match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
		if match:
			try:
				return json.loads(match.group(1))
			except json.JSONDecodeError:
				return None
		return None

# Relevance weights for consensus scoring
RELEVANCE_WEIGHTS = {
	"Highly Relevant": 1.0,
	"Relevant": 0.8,
	"Somewhat Relevant": 0.5,
	"Not Relevant": 0.0
}

class AsyncLLMClient:
	def __init__(self):
		self.client = genai.Client()

	async def generate(self, prompt: str) -> Optional[str]:
		"""
		Generate a response to the prompt using the Google GenAI API.
		"""
		response = await asyncio.to_thread(
			self.client.models.generate_content,
			model="gemini-2.5-flash",
			contents=prompt,
			config=types.GenerateContentConfig(
				response_mime_type="application/json"
			)
		)
		return response.text

class ConsensusEvaluator:
	def __init__(self):
		self.client = AsyncLLMClient()
	
	async def _check_applicability(self, query: str) -> bool:
		"""
		Step 1: Checks whether this query is about scientific consensus.
		"""
		prompt = f"""
		Analyze if this query implies a scientific debate or consensus request.
		Query: "{query}"
		Return JSON: {{ "is_applicable": bool, "reason": "string" }}
		"""
		result_str = await self.client.generate(prompt)
		if result_str is None:
			return False
		result = parse_json_from_text(result_str)
		return result.get("is_applicable", False) if result else False

	async def _analyze_single_paper(self, query: str, paper: Dict) -> Dict:
		"""
		Step 2: Analyze one paper in isolation.
		"""
		prompt = f"""
		Determine the stance of this specific paper regarding the query.
		
		Query: "{query}"
		Paper: {paper['title']}
		Abstract: {paper['abstract']}
		
		Return JSON:
		{{
			"stance": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
			"confidence": float (0.0 to 1.0),
			"reasoning": "1 sentence explanation"
		}}
		"""
		try:
			analysis_str = await self.client.generate(prompt)
			if analysis_str is None:
				return {"paper_id": paper['id'], "error": "No response from API", "stance": "NEUTRAL", "confidence": 0}
			analysis = parse_json_from_text(analysis_str)
			if analysis:
				return {**analysis, "paper_id": paper['id'], "relevance": paper.get("relevance", "Somewhat Relevant")}
			else:
				return {"paper_id": paper['id'], "error": "Failed to parse analysis", "stance": "NEUTRAL", "confidence": 0}
		except Exception as e:
			return {"paper_id": paper['id'], "error": str(e), "stance": "NEUTRAL", "confidence": 0}

	def _calculate_hybrid_score(self, analyses: List[Dict]) -> float:
		"""
		Step 3: Calculate score weighting by Relevance and Confidence.
		"""
		if not analyses:
			return 0.0

		# 1. Extract Stances (x) and Weights (w)
		values = []
		weights = []
		
		for item in analyses:
			# Map stance text to number
			stance_map = {"POSITIVE": 1.0, "NEGATIVE": -1.0, "NEUTRAL": 0.0}
			stance_value = item.get('stance') if isinstance(item, dict) else None
			x = stance_map.get(str(stance_value) if stance_value is not None else "NEUTRAL", 0.0)
			
			# Calculate Weight (Relevance * Confidence)
			# Trust Asta's 'Perfectly Relevant' signal heavily
			relevance_value = item.get('relevance') if isinstance(item, dict) else None
			rel_weight = RELEVANCE_WEIGHTS.get(str(relevance_value) if relevance_value is not None else "Somewhat Relevant", 0.5) 
			conf_weight = item.get('confidence', 0.5) if isinstance(item, dict) else 0.5
			w = rel_weight * conf_weight
			
			values.append(x)
			weights.append(w)

		sum_weights = sum(weights)
		if sum_weights == 0:
			return 0.0

		# 2. Weighted mean (the signal)
		weighted_mean = sum(v * w for v, w in zip(values, weights)) / sum_weights

		# 3. Weighted standard deviation (the noise/controversy)
		# Measures how much the weighted votes deviate from the weighted mean
		variance_numerator = sum(w * ((v - weighted_mean) ** 2) for v, w in zip(values, weights))
		weighted_variance = variance_numerator / sum_weights
		weighted_std_dev = math.sqrt(weighted_variance)

		# 4. The "penalized" score
		# Same logic as your original idea: High variance reduces the final score magnitude
		# We clip the penalty so it doesn't invert the sign
		penalty_factor = max(0.0, 1.0 - weighted_std_dev)
		
		final_score = weighted_mean * penalty_factor
		
		return final_score
	
	async def _generate_synthesis(self, query: str, analyses: List[Dict], consensus_score: float) -> str:
		"""
		Step 4: Generate a summary of the overall scientific consensus.
		"""
		# Build summary of paper analyses for the prompt
		analysis_summary = ""
		for item in analyses:
			paper_id = item.get('paper_id', 'Unknown')
			stance = item.get('stance', 'NEUTRAL')
			reasoning = item.get('reasoning', item.get('error', 'No reasoning provided'))
			analysis_summary += f"Paper {paper_id} [{stance}]: {reasoning}\n"
		
		# Determine consensus label from score
		if abs(consensus_score) > 0.7:
			consensus_label = "strong consensus" if consensus_score > 0 else "strong consensus against"
		elif abs(consensus_score) > 0.3:
			consensus_label = "moderate consensus" if consensus_score > 0 else "moderate consensus against"
		else:
			consensus_label = "contested or uncertain view"
		
		prompt = f"""
		You are a Senior Scientific Editor. Based on the following paper analyses, write a concise 1-2 sentence synthesis summary of the scientific consensus regarding this query.
		
		Query: "{query}"
		Consensus Score: {consensus_score:.3f} (indicates {consensus_label})
		
		Paper Analyses:
		{analysis_summary}
		
		Write a synthesis that:
		1. States the overall consensus status (consensus, contested, or uncertain)
		2. Summarizes the key findings from papers supporting different positions
		3. Is written in a clear, professional scientific tone
		
		Return JSON:
		{{
			"summary": "Your synthesis summary here"
		}}
		"""
		
		try:
			result_str = await self.client.generate(prompt)
			if result_str is None:
				return "Unable to generate synthesis summary."
			result = parse_json_from_text(result_str)
			return result.get("summary", "Unable to generate synthesis summary.") if result else "Unable to generate synthesis summary."
		except Exception as e:
			return f"Error generating synthesis: {str(e)}"
		
	async def process(self, query: str, papers: List[Dict]):
		# Check if the query is a consensus topic
		if not await self._check_applicability(query):
			return {"status": "SKIPPED", "reason": "Not a consensus topic"}

		batch_size = 5
		# Short-circuit check
		batch_1 = papers[:batch_size]
		batch_2 = papers[batch_size:]
		
		# Spawn the analysis tasks

		# Run Batch 1 in parallel
		results = await asyncio.gather(*[self._analyze_single_paper(query, p) for p in batch_1])
		
		# Check for unanimity
		stances = [r['stance'] for r in results]
		if len(set(stances)) == 1 and stances[0] != "NEUTRAL":
			print(f"Unanimous consensus found in top 5. Short-circuiting remaining {len(batch_2)} papers.")
		else:
			# If mixed, process the rest
			if batch_2:
				print("Mixed results. Processing remaining batch...")
				results_2 = await asyncio.gather(*[self._analyze_single_paper(query, p) for p in batch_2])
				results.extend(results_2)

		# Aggregation
		final_score = self._calculate_hybrid_score(results)
		
		# Generate synthesis summary
		print("Generating consensus summary...")
		synthesis = await self._generate_synthesis(query, results, final_score)
		
		return {
			"status": "COMPLETED",
			"consensus_score": final_score,
			"summary": synthesis,
			"breakdown": results
		}


# ==========================================
# DEMO: How this runs in the Asta workflow
# ==========================================
async def main():

	def display_output(insight: Dict):
		if insight.get('status') == 'SKIPPED':
			print("\n[Consensus Evaluation: NOT APPLICABLE]")
			print(f"Reason: {insight.get('reason', 'Unknown')}")
		else:
			print("\n[Consensus Evaluation: APPLICABLE]")
			consensus_score2 = insight.get('consensus_score')
			if consensus_score2 is not None:
				print(f"Consensus Score: {consensus_score2:.3f}")
			print(f"\nSynthesis: {insight.get('summary', 'N/A')}\n")
			print("Detailed Breakdown:")
			for item in insight.get('breakdown', []):
				if isinstance(item, dict):
					stance = item.get('stance', 'NEUTRAL')
					if stance == "POSITIVE":
						icon = "✅"
					elif stance == "NEGATIVE":
						icon = "❌"
					else:
						icon = "➖"
					reasoning = item.get('reasoning', item.get('error', 'Unknown'))
					print(f"   {icon} Paper {item.get('paper_id', 'Unknown')} [{stance}]: {reasoning}")

	scout = ConsensusEvaluator()

	# --- Scenario 1: User asks about Remote Work Productivity ---
	print("Processing Query: 'Does remote work lower productivity?'")
	
	# These are papers Asta (Hypothetically) found:
	retrieved_papers = [
		{"id": "P1", "title": "The WFH Trap", "abstract": "Our study of 500 data entry clerks shows a 10% decrease in throughput when working from home."},
		{"id": "P2", "title": "Creative Freedom", "abstract": "Software engineers reported a 15% increase in code quality and satisfaction when remote."},
		{"id": "P3", "title": "Zoom Fatigue", "abstract": "This paper analyzes the psychological impact of video calls, not direct productivity metrics."},
		{"id": "P4", "title": "Remote Collaboration Costs", "abstract": "Analysis of 2000 knowledge workers reveals a 12% reduction in task completion rates and increased time-to-delivery when working remotely compared to in-office settings."}
	]

	# Run the Evaluator
	insight1 = await scout.process("Does remote work lower productivity?", retrieved_papers)
	# Display the Output
	display_output(insight1)
	
	# --- Scenario 2: Non-applicable query (factual lookup) ---
	print("\n" + "="*70)
	print("Processing Query: 'attention is all you need vaswani 2017'")
	
	lookup_papers = [
		{"id": "P1", "title": "Attention Is All You Need", "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."},
		{"id": "P2", "title": "The Transformer Architecture", "abstract": "This paper reviews the Transformer architecture introduced by Vaswani et al. in 2017 and its impact on NLP."}
	]
	
	insight2 = await scout.process("attention is all you need vaswani 2017", lookup_papers)
	# Display the Output
	display_output(insight2)
	


if __name__ == "__main__":
	asyncio.run(main())