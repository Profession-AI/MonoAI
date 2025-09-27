from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

import asyncio
from contextlib import asynccontextmanager
import json

class McpServer:
    def __init__(
        self,
        name: str,
        server_type: str,               # "http" | "node" | "python"
        server_args: list | None = None,
        server_url: str | None = None,
        env: dict | None = None,
        headers: dict | None = None,    # <-- per HTTP/SSE
        connect_timeout: float = 15.0,  # opzionale
    ):
        self.name = name
        self._server_type = server_type
        self._server_args = server_args or []
        self._server_url = server_url
        self._env = env or {}
        self._headers = headers or {}
        self._session = None
        self._ctx = None
        self._reader = None
        self._writer = None
        self._connect_timeout = connect_timeout

    async def connect(self):
        """Avvia il server MCP e apre la sessione"""
        try:
            if self._server_type == "http":
                if not self._server_url:
                    raise ValueError("URL mancante per server_type='http'")
                
                if self._server_url.endswith("/mcp"):
                    self._ctx = streamablehttp_client(self._server_url, headers=self._headers)
                else:
                    self._ctx = sse_client(self._server_url, headers=self._headers)
            else:
                server_params = StdioServerParameters(
                    command="npx" if self._server_type == "node" else "python",
                    args=self._server_args,
                    env=self._env,
                    stderr="pipe",
                )
                self._ctx = stdio_client(server_params)

            # timeout per non restare appesi
            self._reader, self._writer = await asyncio.wait_for(
                self._ctx.__aenter__(), timeout=self._connect_timeout
            )
            self._session = ClientSession(self._reader, self._writer)
            await self._session.__aenter__()
            await asyncio.wait_for(self._session.initialize(), timeout=self._connect_timeout)

        except Exception as e:
            # prova a chiudere il contesto se parzialmente aperto
            try:
                if self._session:
                    await self._session.__aexit__(None, None, None)
            finally:
                self._session = None
            try:
                if self._ctx:
                    await self._ctx.__aexit__(None, None, None)
            finally:
                self._ctx = None

            # Aggiungi contesto utile all’errore
            where = f"HTTP {self._server_url}" if self._server_type == "http" else f"STDIO args={self._server_args}"
            raise RuntimeError(f"Connessione MCP '{self.name}' fallita ({where}): {e}") from e

    async def get_tools(self):
        if not self._session:
            raise RuntimeError("Non sei connesso. Chiama .connect() prima.")
        resp = await self._session.list_tools()
        return resp.tools

    async def call_tool(self, name: str, args: dict):
        if not self._session:
            raise RuntimeError("Non sei connesso. Chiama .connect() prima.")
        # rimuovi prefisso se presente
        prefix = f"mcp_{self.name}_"
        tool_name = name[len(prefix):] if name.startswith(prefix) else name
        print("CALLING MCP TOOL 2", tool_name, args)
        result = await self._session.call_tool(tool_name, args)
        return {
            "tool_call_id": f"mcp_{self.name}_{tool_name}",
            "role": "tool",
            "name": tool_name,
            "content": json.loads(result.content[0].text),
        }

    async def disconnect(self):
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._ctx:
            await self._ctx.__aexit__(None, None, None)
            self._ctx = None

    @asynccontextmanager
    async def session_context(self):
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()
