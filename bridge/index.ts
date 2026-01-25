
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";


// Create a proxy server
const server = new McpServer({
    name: "farnsworth-proxy",
    version: "1.0.0",
});

// Define tools that mirror the Python server
// This allows Smithery to scan the capabilities
server.tool(
    "farnsworth_remember",
    {
        content: { type: "string", description: "Information to remember" },
        tags: { type: "string", description: "Comma-separated tags" }
    },
    async ({ content, tags }) => {
        // In a real proxy, we'd forward this to the Python process
        return { content: [{ type: "text", text: "Proxied to Python" }] };
    }
);

async function main() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
}

main().catch(console.error);
