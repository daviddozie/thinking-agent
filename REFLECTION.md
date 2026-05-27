# Reflection

## The Architecture Shift

Moving from a local LangChain tool architecture to a decoupled MCP-based architecture changed how the system was designed and managed. In Stage 1, everything existed inside one process, so tools, retrieval, and orchestration could directly share memory and communicate easily. In Stage 2, the MCP server became responsible for domain intelligence while the client focused on orchestration and interaction with the LLM. One operational advantage of this approach was separation of concerns. It became easier to debug issues because server-side retrieval problems and client-side orchestration problems were isolated from each other. The architecture also became more modular and scalable since the server could theoretically be deployed independently from the client.

However, introducing the network boundary also introduced performance overhead. Every MCP interaction required request serialization, transport over HTTP, execution on the server, and response parsing on the client. Compared to local tool execution, this added noticeable latency, especially during reflection workflows involving multiple calls.

## The Sampling Paradox

Implementing MCP Sampling revealed an interesting architectural tradeoff. Instead of the server owning an API key or model instance, the server requests sampling from the client, which already has authenticated LLM access. From a security perspective, this is beneficial because sensitive credentials remain on the client side. Structurally, it also keeps the server lightweight and focused only on domain logic. However, implementing this was challenging because the MCP adapter ecosystem is still evolving, and certain sampling workflows were not fully supported in the installed adapter version. This required restructuring parts of the reflection workflow to keep LLM execution on the client side.

## State & Context Management

Handling hierarchical chunking and multi-query expansion inside the MCP resource significantly changed how context was passed through the system. Since the server behaves as an isolated component, the agent could no longer rely on shared memory between tools. Every resource response needed to contain enough context for downstream reasoning. The multi-query expansion improved retrieval coverage, while the ToT evaluation reduced noisy chunks before they reached the agent. This made context passing more explicit and intentional throughout the architecture.