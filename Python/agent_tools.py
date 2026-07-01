import os

import aiofiles
import aiofiles.os

from datetime import datetime
import dateparser

from langchain_core.tools import BaseTool, tool


MARKDOWN_REPORT_OUTPUT_DIRECTORY = "./outputs/WorkTrackerReports"


def get_all_local_agent_tools() -> list[BaseTool]:
    """
    Returns the list of all the local agent tools.
    """
    return [
        get_relative_timestamp_ms,
        export_markdown_report,
        read_local_notes,
        # TODO: Add more defined tools here
    ]

@tool
def get_relative_timestamp_ms(relative_time_query: str) -> int:
    """
    Calculates the exact UTC millisecond timestamp for a relative time string.
    Always use this before calling tools that require a timestamp!
    Examples of queries: "now", "yesterday at 8am", "last monday", "1st of current month".
    """
    # dateparser safely handles human-readable relative time queries
    parsed_date = dateparser.parse(relative_time_query, settings={'TIMEZONE': 'UTC'})
    if not parsed_date:
        raise ValueError(f"Could not parse time query: '{relative_time_query}'!")
    
    return int(parsed_date.timestamp() * 1000)


@tool
async def export_markdown_report(filename: str, markdown_content: str) -> str:
    """
    Saves a formatted markdown report to the local file system.
    There is a safety measure, which removes any characters, which are not alphabetic, numeric or from the list of allowed special characters [' ', '.', '_', '-'], from the input file name.
    Use this when the user asks to save, export, or write down a summary of their work.
    """
    safe_dir = os.path.expanduser(MARKDOWN_REPORT_OUTPUT_DIRECTORY)
    
    # Asynchronous directory creation
    await aiofiles.os.makedirs(safe_dir, exist_ok=True)
    
    # Clean the filename to prevent path traversal or invalid characters
    safe_filename = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).rstrip()
    if not safe_filename.endswith(".md"):
        safe_filename += ".md"
        
    filepath = os.path.join(safe_dir, safe_filename)
    
    # Asynchronous file writing
    async with aiofiles.open(filepath, mode="w", encoding="utf-8") as f:
        await f.write(markdown_content)
        
    return f"Successfully saved markdown report to {filepath}"


@tool
async def read_local_notes(filepath: str) -> str:
    """
    Reads the content of a local text or markdown file. 
    Use this when the user asks you to read meeting notes or a local to-do list to create tasks from it.
    """
    expanded_path = os.path.expanduser(filepath)
    
    try:
        # Asynchronously attempt to open and read the file
        async with aiofiles.open(expanded_path, mode="r", encoding="utf-8") as f:
            content = await f.read()
            return content
            
    except FileNotFoundError:
        # This error string is safely returned to the LLM so it knows it failed
        return f"Error: File not found at {expanded_path}. Ask the user to verify the path."
    except PermissionError:
        return f"Error: You do not have permission to read the file at {expanded_path}."
    except Exception as e:
        return f"Error reading file: {str(e)}"
