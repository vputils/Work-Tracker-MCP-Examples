import * as fs from 'fs/promises';
import * as path from 'path';
import * as chrono from 'chrono-node';
import { tool } from '@langchain/core/tools';
import { z } from 'zod'; 
import { expandHomeDir } from './utils.js';
import { StructuredTool } from "@langchain/core/tools";
import dedent from 'dedent';

const MARKDOWN_REPORT_OUTPUT_DIRECTORY = "~/WorkTrackerReports";

const SAFE_FILENAME_REGEX = /[^a-zA-Z0-9 ._-]/g;

/**
 * The tool for calculating exact UTC millisecond timestamp 
 * from a human-readable relative time string.
 */
export const getRelativeTimestampMs: StructuredTool = tool(
    async ({ relativeTimeQuery }) => {
        const parsedDate = chrono.parseDate(relativeTimeQuery);
        
        if (!parsedDate) {  // Parsing failed and it returned null
            // Throwing an Error here will be caught by the Promise.allSettled logic
            throw new Error(`Could not parse time query: '${relativeTimeQuery}'!`);
        }

        // Return as string to avoid precision loss or formatting issues
        return parsedDate.getTime().toString();
    },
    {
        name: "get_relative_timestamp_ms",
        description: dedent(`
                Calculates the exact UTC millisecond timestamp for a relative time string.
                Always use this before calling tools that require a timestamp!
                Examples of queries: "now", "yesterday at 8am", "last monday", "1st of current month".
            `.trim()),
        schema: z.object({
            relativeTimeQuery: z.string().describe(
                "The human-readable time string to parse. Examples: 'now', 'yesterday at 8am', 'last monday', '1st of current month'."),
        }),
    }
);

/**
 * The tool for exporting of Markdown content into a file.
 */
export const exportMarkdownReport: StructuredTool = tool(
    async ({ filename, markdownContent }) => {
        const safeDir = expandHomeDir(MARKDOWN_REPORT_OUTPUT_DIRECTORY);

        await fs.mkdir(safeDir, { recursive: true });

        // Clean the filename to prevent path traversal or invalid characters
        let safeFilename = filename.replace(SAFE_FILENAME_REGEX, "").trimEnd();
        if (!safeFilename.endsWith(".md")) {
            safeFilename += ".md";
        }

        const filepath = path.join(safeDir, safeFilename);
        await fs.writeFile(filepath, markdownContent);

        return `Successfully saved markdown report to ${filepath}`;
    },
    {
        name: "export_markdown_report",
        description: dedent(`
                Saves a formatted markdown report to the local file system. 
                Use this when the user asks to save, export, or write down a summary of their work.
            `).trim(),
        schema: z.object({
            filename: z.string().describe(
                "The requested file name."
                + " There is a safety measure, which removes any characters,"
                + " which are not alphabetic, numeric or from the list of allowed special characters [' ', '.', '_', '-'],"
                + " from the input file name."),
            markdownContent: z.string().describe("The markdown content to be written."),
        }),
    }
)

/**
 * The tool for reading a local file (i.e. notes).
 */
export const readLocalNotes: StructuredTool = tool(
    async ({ filepath }) => {
        const expandedPath = expandHomeDir(filepath);

        try {
            const content = await fs.readFile(expandedPath, "utf-8");
            return content;
        } catch (error: unknown) {
            const exception = error as NodeJS.ErrnoException;
            const errorCode = exception.code as string; 
            switch (errorCode) {
                // "ENOENT" is Node.js's standard "Error NO ENTry" (File not found)
                case 'ENOENT':
                    return `Error: File not found at ${expandedPath}. Ask the user to verify the path.`;
                case 'EACCESS':
                    return `Error: You do not have permission to read the file at ${expandedPath}.`;
                default:
                    return `Error reading file: ${exception.message}`;
            }
        }
    }, {
        name: "read_local_notes",
        description: dedent(`
                Reads the content of a local text or markdown file. 
                Use this when the user asks you to read meeting notes or a local to-do list to create tasks from it.
            `).trim(),
        schema: z.object({
            filepath: z.string().describe("The path to the local file.")
        }),
    }
)

/**
 * Get the list of all the existing local tools
 * @returns The list of all local tools
 */
export function getAllLocalAgentTools(): StructuredTool[] {
    return [
        getRelativeTimestampMs,
        exportMarkdownReport,
        readLocalNotes,
        // TODO: Add more defined tools here 
    ];
}
