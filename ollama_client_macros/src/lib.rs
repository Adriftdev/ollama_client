extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields, Meta};

#[proc_macro_derive(OllamaTool, attributes(doc))]
pub fn ollama_tool_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    
    // Convert PascalCase to snake_case for the tool name
    let name_str = name.to_string();
    let mut snake_case_name = String::new();
    for (i, c) in name_str.chars().enumerate() {
        if c.is_uppercase() {
            if i != 0 {
                snake_case_name.push('_');
            }
            snake_case_name.push(c.to_ascii_lowercase());
        } else {
            snake_case_name.push(c);
        }
    }

    let struct_doc = extract_doc_comment(&input.attrs).unwrap_or_default();
    
    // Parse fields
    let mut properties = quote! {};
    let mut required_fields: Vec<String> = Vec::new();
    
    if let Data::Struct(data) = &input.data {
        if let Fields::Named(fields) = &data.fields {
            for field in &fields.named {
                let field_name = field.ident.as_ref().unwrap();
                let field_name_str = field_name.to_string();
                let field_doc = extract_doc_comment(&field.attrs).unwrap_or_default();
                
                let ty = &field.ty;
                let type_str = quote!(#ty).to_string();
                let type_str = type_str.replace(" ", "");
                
                let (json_type, is_optional) = map_type_to_json_schema(&type_str);
                
                if !is_optional {
                    required_fields.push(field_name_str.clone());
                }
                
                properties = quote! {
                    #properties
                    map.insert(#field_name_str.to_string(), ::serde_json::json!({
                        "type": #json_type,
                        "description": #field_doc
                    }));
                };
            }
        }
    }
    
    let required_array = if required_fields.is_empty() {
        quote! { ::serde_json::json!([]) }
    } else {
        quote! { ::serde_json::json!([ #(#required_fields),* ]) }
    };

    let expanded = quote! {
        impl $crate::tools::OllamaTool for #name {
            fn name(&self) -> &'static str {
                #snake_case_name
            }
            
            fn tool_definition(&self) -> ::serde_json::Value {
                let mut map = ::serde_json::Map::new();
                #properties
                
                ::serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": #snake_case_name,
                        "description": #struct_doc,
                        "parameters": {
                            "type": "object",
                            "properties": map,
                            "required": #required_array
                        }
                    }
                })
            }
            
            fn execute_from_json(&self, args: ::serde_json::Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
                let parsed: Self = ::serde_json::from_value(args)?;
                // The user must implement a separate `execute` method or trait, 
                // but the specification requires `execute_from_json` to do the dynamic routing.
                // Assuming `self.run()` exists on the struct:
                parsed.run()
            }
        }
    };
    
    TokenStream::from(expanded)
}

fn extract_doc_comment(attrs: &[syn::Attribute]) -> Option<String> {
    let mut docs = Vec::new();
    for attr in attrs {
        if attr.path().is_ident("doc") {
            if let Meta::NameValue(mnv) = &attr.meta {
                if let syn::Expr::Lit(expr_lit) = &mnv.value {
                    if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                        docs.push(lit_str.value().trim().to_string());
                    }
                }
            }
        }
    }
    if docs.is_empty() {
        None
    } else {
        Some(docs.join(" "))
    }
}

fn map_type_to_json_schema(ty: &str) -> (&'static str, bool) {
    if ty.starts_with("Option<") {
        let inner = &ty[7..ty.len()-1];
        return (map_primitive(inner), true);
    }
    (map_primitive(ty), false)
}

fn map_primitive(ty: &str) -> &'static str {
    match ty {
        "String" | "&str" => "string",
        "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "usize" | "isize" => "integer",
        "f32" | "f64" => "number",
        "bool" => "boolean",
        _ => "string", // default fallback
    }
}
