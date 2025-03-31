class LlmModelException(Exception):
    def __init__(self, message="No models available"):
        super().__init__(message)
