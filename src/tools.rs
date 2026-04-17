use serde_json::Value;
use std::collections::HashMap;

/// A trait for generating JSON Schemas and executing logic for Ollama tools.
/// 
/// This trait is typically derived using `#[derive(OllamaTool)]`. 
/// The derived struct must implement a `run(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>>` method.
pub trait OllamaTool: Send + Sync {
    /// The snake_case identifier of the tool.
    fn name(&self) -> &'static str;
    
    /// Returns the complete JSON schema required by the Ollama API.
    fn tool_definition(&self) -> Value;
    
    /// Deserializes the model's argument payload and executes the underlying logic.
    fn execute_from_json(&self, args: Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
}

/// A registry for dynamically routing tool calls to their respective implementations.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn OllamaTool>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Registers a tool into the map using its `name()` as the key.
    pub fn register<T: OllamaTool + 'static>(&mut self, tool: T) {
        self.tools.insert(tool.name().to_string(), Box::new(tool));
    }

    /// Iterates over values and calls `tool_definition()`.
    pub fn get_definitions(&self) -> Vec<Value> {
        self.tools.values().map(|t| t.tool_definition()).collect()
    }

    /// Performs a `HashMap::get`, returning an error string if the tool is not found,
    /// or executing the dynamic trait object.
    pub fn execute(&self, name: &str, args: Value) -> Result<String, String> {
        if let Some(tool) = self.tools.get(name) {
            tool.execute_from_json(args).map_err(|e| e.to_string())
        } else {
            Err(format!("Tool not found: {}", name))
        }
    }
}
