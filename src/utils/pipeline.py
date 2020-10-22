class Pipe:
    
    def __init__(self, functions=None):
        self.functions = functions
        
    def call_pipeline(self, *args):
        """
        This function currently only supports one argument input.
        
        @param function_pipline: a list of functions: [func1, func2, func3...]
        Note: func1 output is the input for func2 etc
        """
        
        result = args[0]
        for f in self.functions:
            result = f(result)

        return result