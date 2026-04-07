"""OpenAI LLM provider implementation."""

from openai import OpenAI, NOT_GIVEN
from typing import Dict, Any, Callable, Optional, List, Union
from .base.base_llm_provider import BaseLLMProvider, ToolExecutionError
from modules.guardrails.content_safety_decorator import with_content_safety


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider using Responses API."""
    
    provider_name = "openai"
    
    def __init__(self, api_key: str, models_config: Optional[List[Dict[str, Any]]] = None):
        """Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            models_config: Optional list of model configuration dictionaries
        """
        super().__init__(api_key, models_config=models_config)
        self.client = OpenAI(api_key=api_key)
        self.input: List[Any] = []  # Track entire conversation history
    
    def _create_message_content(self, user_prompt: str, image_path: Optional[str] = None) -> Union[str, List[Dict[str, Any]]]:
        """
        Create message content for OpenAI Responses API format.
        
        Args:
            user_prompt: The text prompt from the user
            image_path: Optional path to local image file
            
        Returns:
            Either a string (text only) or a list of content objects (text + image)
        """
        if image_path is None:
            return user_prompt
        
        # Encode the image
        base64_image = self._encode_image_to_base64(image_path)
        
        # Determine the MIME type based on file extension
        from pathlib import Path
        image_file = Path(image_path)
        mime_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_type_map.get(image_file.suffix.lower(), 'image/jpeg')
        
        # Create content array with text and image for Responses API
        return [
            {
                "type": "input_text",
                "text": user_prompt
            },
            {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{base64_image}",
                "detail": "auto"
            }
        ]
    
    @with_content_safety
    def invoke(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: Optional[float] = 0.1,
        model: Optional[str] = None,
        tools: bool = False,
        funcs: Optional[List[Callable]] = None,
        tool_choice: Union[str, Dict[str, Any]] = 'auto',
        parse: bool = False,
        parser: Optional[dict] = None,
        standard_output: bool = True,
        max_tool_calls: Optional[int] = 3,
        image_path: Optional[str] = None
    ):
        """
        Invoke the OpenAI model with support for iterative tool calling and vision.
        
        Args:
            user_prompt: The user's prompt
            system_prompt: The system prompt/instructions
            temperature: Sampling temperature (0.0 to 1.0)
            model: The model to use (e.g., "gpt-4.1")
            tools: Whether to enable tools/function calling
            funcs: List of callable functions available to the model
            tool_choice: Tool choice strategy ('auto', 'required', 'none')
            parse: Whether to parse the response with a custom parser
            parser: Parser configuration dictionary
            standard_output: Whether to return standardized output format
            max_tool_calls: Maximum number of tool call iterations (default: 3)
            image_path: Optional path to local image file for vision capabilities
            
        Returns:
            Either raw response object or standardized dictionary with content, tool_calls, and token_usage
        """
        # Resolve model if not provided
        if not model:
            model = self.get_default_model()
        
        response = None
        
        if tools:
            if funcs is None:
                raise ValueError("Tools definition must be provided when tools is True")
            
            response = self._handle_tool_calling(
                user_prompt, system_prompt, temperature, model, funcs, tool_choice, max_tool_calls, image_path
            )
        else:
            # For non-tool calling, add messages to conversation history
            message_content = self._create_message_content(user_prompt, image_path)
            input_list = [
                {"role": "user", "content": message_content}
            ]
            self.input.extend(input_list)
            
            response = self.client.responses.create(
                model=model or "gpt-4.1",
                temperature=temperature,
                instructions=system_prompt,
                input=self.input,
            )
            
            # Add response to conversation history
            if hasattr(response, 'output'):
                self.input.extend(response.output)
        
        # Ensure response is not None
        if response is None:
            raise RuntimeError("Failed to get response from OpenAI")
        
        if not standard_output:
            return response
        
        # Handle parsing if requested
        if parse:
            return self._handle_parsing(response, parser)
        
        # Extract tool calls from response if any
        tool_calls_in_response = None
        if hasattr(response, 'output'):
            tool_calls_in_response = [item for item in response.output if hasattr(item, 'type') and item.type == "function_call"]
        
        # Extract text content using the response.output_text property
        content_text = self._extract_response_text(response)
        
        return {
            "content": content_text,
            "tool_calls": tool_calls_in_response,
            "token_usage": self._extract_token_usage(response)
        }
    
    def _handle_tool_calling(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: Optional[float],
        model: Optional[str],
        funcs: List[Callable],
        tool_choice: Union[str, Dict[str, Any]],
        max_tool_calls: Optional[int],
        image_path: Optional[str] = None
    ):
        """Handle iterative tool calling logic."""
        # Create tools
        tools_def = []
        for func in funcs:
            tools_def.append(self.create_tool(func))
        
        # Add initial messages to conversation history
        message_content = self._create_message_content(user_prompt, image_path)
        input_list = [
            {"role": "user", "content": message_content}
        ]
        self.input.extend(input_list)
        
        # Track tool call iterations
        tool_call_count = 0
        max_iterations = max_tool_calls or 3
        response = None
        
        while tool_call_count < max_iterations:
            response = self.client.responses.create(
                model=model or "gpt-4.1",
                temperature=temperature,
                input=self.input,
                instructions=system_prompt,
                tools=tools_def,
                tool_choice=tool_choice if tool_choice is not None else NOT_GIVEN  # type: ignore
            )
            
            # Add the model's response to the conversation
            self.input.extend(response.output)
            
            # Check if there are any tool calls in the response
            tool_calls = [item for item in response.output if hasattr(item, 'type') and item.type == "function_call"]
            
            if not tool_calls:
                # No tool calls, we're done
                break
            
            # Execute each tool call and add results to conversation
            for tool_call in tool_calls:
                try:
                    tool_result = self.execute_tool(tool_call, funcs)
                    
                    # Add tool output to conversation
                    self.input.append({
                        "type": "function_call_output",
                        "call_id": getattr(tool_call, 'call_id', ''),
                        "output": tool_result
                    })
                except ToolExecutionError as e:
                    self.input.append({
                        "type": "function_call_output",
                        "call_id": getattr(tool_call, 'call_id', ''),
                        "output": f"Error: {str(e)}"
                    })
            
            tool_call_count += 1
        
        # If we reached max iterations, get final response without tools
        if tool_call_count >= max_iterations:
            response = self.client.responses.create(
                model=model or "gpt-4.1",
                temperature=temperature,
                input=self.input
            )
            
            # Add final response to conversation history
            if hasattr(response, 'output'):
                self.input.extend(response.output)
        
        # Ensure we have a response
        if response is None:
            raise RuntimeError("Failed to get response from tool calling")
        
        return response
    
    def create_tool(self, func: Callable, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a tool definition in OpenAI Responses API format.
        
        Args:
            func: The callable function to convert to a tool definition
            tool_name: Optional name override (defaults to function name)
            
        Returns:
            Tool definition dictionary in OpenAI format
        """
        import inspect
        
        function_name = tool_name or func.__name__
        
        # Get function signature and docstring
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or f"Function {function_name}"
        
        # Build parameters schema with strict mode requirements
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name in ['self', 'cls']:
                continue
            
            # Determine type from annotation or default to string
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "number"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"
            
            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}"
            }
            
            # Add to required if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        # Create tool definition following latest OpenAI Responses API format
        tool_def = {
            "type": "function",
            "name": function_name,
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False  # Required for strict mode
            },
            "strict": True
        }
        
        return tool_def
    
    def execute_tool(self, tool_call: Any, available_tools: List[Callable]) -> Any:
        """
        Execute a tool call from the OpenAI Responses API format.
        
        Args:
            tool_call: The tool call object from OpenAI Responses API
            available_tools: List of callable functions available to the model
            
        Returns:
            Result as a string or JSON-encoded string
        """
        import json
        
        func_name = getattr(tool_call, 'name', None)
        if not func_name:
            raise ToolExecutionError("unknown", "Function name not found in tool call")
        
        try:
            # Parse arguments from the tool call
            arguments = getattr(tool_call, 'arguments', None)
            if arguments is None:
                args = {}
            elif isinstance(arguments, str):
                args = json.loads(arguments)
            elif isinstance(arguments, dict):
                args = arguments
            else:
                args = {}
            
            # Find the matching function by name
            func = None
            for f in available_tools:
                if f.__name__ == func_name:
                    func = f
                    break
            if func is None:
                raise ToolExecutionError(func_name, f"Function '{func_name}' not found in available_tools")
            
            # Execute the function with the parsed arguments
            result = func(**args)
            
            # Return result as string or JSON-encoded string
            if isinstance(result, (dict, list)):
                return json.dumps(result)
            else:
                return str(result)
        
        except Exception as e:
            raise ToolExecutionError(func_name, str(e), e)
    
    def _extract_response_text(self, response: Any) -> str:
        """Extract text content from OpenAI Responses API response."""
        if hasattr(response, 'output_text'):
            return response.output_text
        return str(response)
    
    def _extract_token_usage(self, response: Any) -> Optional[Any]:
        """Extract token usage from OpenAI Responses API response."""
        return getattr(response, 'usage', None)
    
    def create_embedding(self, text: List[str], model: str):
        """
        Create an embedding for a given text using the specified model.
        
        Args:
            text: List of text strings to embed
            model: Embedding model to use
            
        Returns:
            Embedding response from OpenAI
        """
        response = self.client.embeddings.create(
            input=text,
            model=model
        )
        return response
