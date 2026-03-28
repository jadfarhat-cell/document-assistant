from typing import Generator, Optional


class OllamaClient:
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Always cite your sources by referencing the [Source X] markers in your response.
If the context doesn't contain relevant information, say so clearly.
Be concise and accurate."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import ollama
            except ImportError:
                raise ImportError("ollama is required. Install with: pip install ollama")
            self._client = ollama.Client(host=self.base_url)
        return self._client

    def generate(
        self,
        query: str,
        context: str,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        prompt = self._build_prompt(query, context)
        if stream:
            return self._stream_response(prompt)
        else:
            return self._generate_response(prompt)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""Context:
{context}

Question: {query}

Answer based on the context above. Cite sources using [Source X] notation."""

    def _generate_response(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]

    def _stream_response(self, prompt: str) -> Generator[str, None, None]:
        stream = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]

    def list_models(self) -> list[str]:
        try:
            models = self.client.list()
            return [m["name"] for m in models.get("models", [])]
        except Exception:
            return []

    def check_connection(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False
