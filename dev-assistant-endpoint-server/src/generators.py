from vllm import LLM, SamplingParams


class GeneratorBase:
    """
    Base class for all generators.

    Defines the interface for generating code from a query and parameters.
    """

    def generate(self, query: str, parameters: dict) -> str:
        """
        Generates code from the given query and parameters.

        Args:
            query (str): The query to generate code from.
            parameters (dict): The parameters to use for generating the code.

        Returns:
            str: The generated code.
        """
        raise NotImplementedError

    def __call__(self, query: str, parameters: dict = None) -> str:
        """
        Generates code from the given query and parameters.

        Args:
            query (str): The query to generate code from.
            parameters (dict): The parameters to use for generating the code.

        Returns:
            str: The generated code.
        """
        return self.generate(query, parameters)


class LuaCoder(GeneratorBase):
    """
    A class that generates Lua code using a pre-trained language model.

    Attributes:
    -----------
    llm : LLM
        The pre-trained language model used for generating Lua code.
    sampling_params : SamplingParams
        The sampling parameters used for generating Lua code.

    Methods:
    --------
    generate(query: str, parameters: dict) -> str:
        Generates Lua code given a query and parameters.
    """

    def __init__(self):
        model_path = "./models/luacoder"
        self.llm = LLM(model=model_path)
        self.sampling_params = SamplingParams(temperature=0.2, top_p=0.95, top_k=4, max_tokens=128)

    def generate(self, query: str, parameters: dict) -> str:
        """
        Generates Lua code given a query and parameters.

        Parameters:
        -----------
        query : str
            The query used for generating Lua code.
        parameters : dict
            The parameters used for generating Lua code.

        Returns:
        --------
        str
            The generated Lua code.
        """
        if query == "":
            return ""
        import time
        start_time = time.time()
        outputs = self.llm.generate(query, self.sampling_params)
        # Stop the timer
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        generated_text = outputs[0].outputs[0].text
        token_ids = outputs[0].outputs[0].token_ids
        print(f'Generate token: {len(token_ids)/elapsed_time:.2f} token/s')
        return generated_text.strip().replace('\r','')
    