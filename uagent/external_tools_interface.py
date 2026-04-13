from abc import ABC, abstractmethod
from typing import Any, Dict


class ExternalToolProvider(ABC):
    @abstractmethod
    def run_tool(self, tool_name: str, image_path: str, prompt: str = "") -> Dict[str, Any]:
        raise NotImplementedError


class DummyExternalToolProvider(ExternalToolProvider):
    def run_tool(self, tool_name: str, image_path: str, prompt: str = "") -> Dict[str, Any]:
        return {
            "tool_name": tool_name,
            "status": "ok",
            "output": f"[dummy output] tool={tool_name} image={image_path}",
            "metadata": {
                "prompt": prompt,
                "provider": "dummy",
            },
        }


def normalize_external_result(tool_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tool_name": result.get("tool_name", tool_name),
        "status": result.get("status", "ok"),
        "output": result.get("output", ""),
        "metadata": result.get("metadata", {}),
    }
