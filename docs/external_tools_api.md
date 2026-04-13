# External Tools API

This document describes the intended interface for connecting external domain tools to UAgent.

## Goal

The research codebase contains several task-specific tools embedded directly in the implementation. For easier reuse, the packaged `UAgent` directory exposes a clean provider interface.

## Core Interface

External tools should implement the provider contract in:

- `uagent/external_tools_interface.py`

## Required Method

```python
run_tool(tool_name: str, image_path: str, prompt: str = "") -> dict
```

## Expected Return Format

```python
{
    "tool_name": "Satellite Image Land Use Inference Tool",
    "status": "ok",
    "output": "Predominantly residential with sparse commercial strips.",
    "metadata": {
        "provider": "your-provider-name"
    }
}
```

## Intended Use

1. The pipeline requests a tool by name.
2. Your provider resolves that tool name to an implementation.
3. The provider returns a normalized dictionary.
4. The pipeline can then pack that output into execution records and urban-state construction.
