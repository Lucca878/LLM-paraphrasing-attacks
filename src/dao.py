class AttemptResult:
    """Result of a single paraphrase attempt within one attack sequence."""

    def __init__(self, text: str, label: int, confidence: float, duration_ms: int, prompt: str, length_reprompt: str = ""):
        self.text = text
        self.label = label            # 0 = deceptive, 1 = truthful
        self.confidence = confidence  # percentage (0–100)
        self.duration_ms = duration_ms
        self.prompt = prompt
        self.length_reprompt = length_reprompt  # last reprompt used if word limit was violated, else ""


class AttackSequence:
    """All data for one complete paraphrasing attack sequence (one statement × one LLM run)."""

    def __init__(
        self,
        session_id: str,
        statement_id: int,
        original_text: str,
        original_label: int,       # 0 = deceptive, 1 = truthful
        original_confidence: float,  # percentage (0–100)
        llm_architecture: str,
        temperature: float,
    ):
        self.session_id = session_id
        self.statement_id = statement_id
        self.original_text = original_text
        self.original_label = original_label
        self.original_confidence = original_confidence
        self.llm_architecture = llm_architecture
        self.temperature = temperature
        self.session_start = ""
        self.session_end = ""
        self.total_duration_ms = 0
        self.attempts: list = []   # list[AttemptResult]
        self.strategies = ""
        self.strategy_prompt = ""
