# consensus-evaluator

Consensus Evaluator is an agentic module designed to integrate into the Asta "Paper Finder" pipeline. While LLMs successfully automate the synthesis of academic literature, the resulting efficiency can inadvertently remove the "contextual intuition" a human researcher gains from manual discovery. This module ensures users aren't presented with a "false sense of certainty" by explicitly surfacing the scientific consensus—or lack thereof—within the retrieved papers.

### Key Features
- Instead of a single synthesized answer, the scout provides a Consensus Summary that paints the "complete picture" of the field’s current stance.
- Routing agent first determines if the user's query is a subject of scientific debate, ensuring the transparency layer only triggers when epistemically appropriate.
- Built with asyncio to parallelize paper analysis, maintaining Asta’s high efficiency while adding deeper insight.
- Calculates a final score (S) by combining weighted mean stance and weighted standard deviation, providing a quantitative measure of scientific certainty.