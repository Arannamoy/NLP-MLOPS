### MCP - Model Context Protocol
```mermaid
graph TB;
 A[MCP HOST. IDE,Cloud Desktop, Cursor IDE, AApplication] --> B[MCP Server 0. Code Repo]
 A[MCP HOST. IDE,Cloud Desktop, Cursor IDE, AApplication] --> C[MCP Server 1. RAG DB]
 A[MCP HOST. IDE,Cloud Desktop, Cursor IDE, AApplication] --> D[MCP Server 2. API's]
 A[MCP HOST. IDE,Cloud Desktop, Cursor IDE, AApplication] --> E[MCP Server 3. Duck Duck Go Search]
```

### Communication betwen the component

```mermaid
graph TD;
    A[I/P MCP Host] --> B[LLM]
    B[LLM] --> A[I/P MCP Host]
    A[I/P MCP Host] --> C[MCP Servers]
    C[MCP Servers] --> A[I/P MCP Host] 
    
```