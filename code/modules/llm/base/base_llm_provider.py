"""Abstract base class for all LLM providers."""

import json
import inspect
import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, TypedDict, Any, Callable, Optional, List, Union
from ..response_parsers.xml_parser import XmlResponse


class LLMResponse(TypedDict):
    """Standardized LLM response format."""
    content: Any
    tool_calls: Optional[list[Any]]
    token_usage: Any


class ToolExecutionError(Exception):
    """Exception raised when tool execution fails."""
    
    def __init__(self, tool_name: str, message: str, original_error: Optional[Exception] = None):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' execution failed: {message}")


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    # Class attribute to be set by implementations
    provider_name: str
    
    def __init__(self, api_key: str, models_config: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None):
        """Initialize LLM provider.
        
        Args:
            api_key: API key for the LLM provider
            models_config: Optional list of model configuration dictionaries or a single model dict.
                          Each dict should have 'model_name' and other model properties.
                          Can be:
                          - A list of model dicts (for backward compatibility)
                          - A single model dict (new format from config.py)
                          - A dict already keyed by model_name
        """
        self.api_key = api_key
        
        # Store models config as a dictionary keyed by model_name for O(1) lookup
        if models_config is None:
            self.models_config: Dict[str, Dict[str, Any]] = {}
        elif isinstance(models_config, list):
            # Convert list to dict keyed by model_name
            self.models_config = {
                model['model_name']: model 
                for model in models_config 
                if 'model_name' in model
            }
        elif isinstance(models_config, dict):
            # Check if it's already keyed by model_name or a single model dict
            if 'model_name' in models_config:
                # Single model dict - wrap it
                model_name = models_config['model_name']
                self.models_config = {model_name: models_config}
            else:
                # Already keyed by model_name
                self.models_config = models_config
        else:
            self.models_config: Dict[str, Dict[str, Any]] = {}
    
    @abstractmethod
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
    ) -> Union[Dict[str, Any], Any]:
        """
        Invoke the LLM model with support for iterative tool calling and vision.
        
        Args:
            user_prompt: The user's prompt
            system_prompt: The system prompt/instructions
            temperature: Sampling temperature (0.0 to 1.0)
            model: The model to use
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
        pass
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode a local image file to base64 string.
        
        Args:
            image_path: Path to the local image file
            
        Returns:
            Base64 encoded string of the image
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the file is not a supported image format
        """
        image_file = Path(image_path)
        
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check if file has a supported image extension
        supported_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        if image_file.suffix.lower() not in supported_extensions:
            raise ValueError(f"Unsupported image format: {image_file.suffix}. Supported formats: {supported_extensions}")
        
        try:
            with open(image_path, "rb") as image_file_obj:
                image_data = image_file_obj.read()
                return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to read and encode image: {str(e)}")
    
    def _create_message_content(self, user_prompt: str, image_path: Optional[str] = None) -> Union[str, List[Dict[str, Any]]]:
        """
        Create message content that can include both text and image.
        
        This is a base implementation that can be overridden by providers
        if they need different message content formats.
        
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
        image_file = Path(image_path)
        mime_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_type_map.get(image_file.suffix.lower(), 'image/jpeg')
        
        # Base implementation - providers can override for their specific format
        return [
            {
                "type": "text",
                "text": user_prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            }
        ]
    
    def create_tool(self, func: Callable, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a tool definition from a callable function.
        
        This is a base implementation that can be overridden by providers
        if they need different tool definition formats.
        
        Args:
            func: The callable function to convert to a tool definition
            tool_name: Optional name override (defaults to function name)
            
        Returns:
            Tool definition dictionary
        """
        function_name = tool_name or func.__name__
        
        # Get function signature and docstring
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or f"Function {function_name}"
        
        # Build parameters schema
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
        
        # Base tool definition format - providers can override
        tool_def = {
            "type": "function",
            "function": {
                "name": function_name,
                "description": doc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        
        return tool_def
    
    def execute_tool(self, tool_call: Any, available_tools: List[Callable]) -> Any:
        """
        Execute a tool call.
        
        This is a base implementation that can be overridden by providers
        if they have different tool call formats.
        
        Args:
            tool_call: The tool call object from the LLM response
            available_tools: List of callable functions available to the model
            
        Returns:
            Result as a string or JSON-encoded string
            
        Raises:
            ToolExecutionError: If tool execution fails
        """
        # Try to get function name from different possible attributes
        func_name = None
        if hasattr(tool_call, 'name'):
            func_name = tool_call.name
        elif hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
            func_name = tool_call.function.name
        
        if not func_name:
            raise ToolExecutionError("unknown", "Function name not found in tool call")
        
        try:
            # Parse arguments from the tool call
            arguments = None
            if hasattr(tool_call, 'arguments'):
                arguments = tool_call.arguments
            elif hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                arguments = tool_call.function.arguments
            
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
    
    def _handle_parsing(self, response: Any, parser: Optional[dict]) -> Dict[str, Any]:
        """
        Handle response parsing logic.
        
        Args:
            response: The raw response from the LLM provider
            parser: Parser configuration dictionary
            
        Returns:
            Parsed response in standardized format
            
        Raises:
            ValueError: If parser is not provided or unsupported parser type
        """
        if parser is None:
            raise ValueError("Parser must be provided when parse is True")
        
        if parser['type'] == 'xml':
            result = {}
            
            # Extract text content from response - providers should override _extract_response_text
            response_text = self._extract_response_text(response)
            
            for tag in parser['args']['tag']:
                o_tag = '<' + tag + '>'
                c_tag = '</' + tag + '>'
                xml_response, isvalid = XmlResponse.extract_multiple(
                    response_text, o_tag, c_tag
                )
                if isvalid:
                    for res in xml_response:
                        if tag not in result:
                            result[tag] = []
                        try:
                            parsed_content = json.loads(res)
                            result[tag].append(parsed_content)
                        except:
                            result[tag].append(res)
            
            return {
                "content": result,
                "tool_calls": None,
                "token_usage": self._extract_token_usage(response)
            }
        else:
            raise ValueError("Unsupported parser type")
    
    def _extract_response_text(self, response: Any) -> str:
        """
        Extract text content from provider-specific response object.
        
        Providers should override this method if their response format differs.
        
        Args:
            response: The raw response from the LLM provider
            
        Returns:
            Text content as string
        """
        # Default implementation - try common attributes
        if hasattr(response, 'output_text'):
            return response.output_text
        elif hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message'):
                return response.choices[0].message.content or ""
        return str(response)
    
    def _extract_token_usage(self, response: Any) -> Optional[Any]:
        """
        Extract token usage from provider-specific response object.
        
        Providers should override this method if their response format differs.
        
        Args:
            response: The raw response from the LLM provider
            
        Returns:
            Token usage information or None
        """
        # Default implementation - try common attributes
        if hasattr(response, 'usage'):
            return response.usage
        return None
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get properties for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary or None if model not found
        """
        return self.models_config.get(model_name)
    
    def list_models(self) -> List[str]:
        """
        List all available model names.
        
        Returns:
            List of model names
        """
        return list(self.models_config.keys())
    
    def get_model_property(self, model_name: str, property_name: str) -> Optional[Any]:
        """
        Get specific property value for a model.
        
        Args:
            model_name: Name of the model
            property_name: Name of the property to retrieve
            
        Returns:
            Property value or None if model or property not found
        """
        model_info = self.get_model_info(model_name)
        if model_info:
            return model_info.get(property_name)
        return None
    
    def has_model(self, model_name: str) -> bool:
        """
        Check if model exists in configuration.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model exists, False otherwise
        """
        return model_name in self.models_config
    
    def get_default_model(self) -> Optional[str]:
        """
        Get default model name (first model in config or provider-specific).
        
        Returns:
            Default model name or None if no models configured
        """
        if self.models_config:
            # Return first model name
            return next(iter(self.models_config.keys()))
        return None

