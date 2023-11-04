/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unsafe-call */
/* eslint-disable @typescript-eslint/restrict-template-expressions */

import * as vscode from "vscode";
import * as path from 'path';
import * as fs from 'fs'
import fetch from "node-fetch";
import { WorkspaceConfiguration, workspace} from "vscode";
import type { Config as HFCodeConfig } from "./configTemplates"



export default function openPanel(context: vscode.ExtensionContext) {
    const panel = vscode.window.createWebviewPanel(
        'DevAssistantChat', 
        'DevAssistantChat', 
        vscode.ViewColumn.One, 
        {
            enableScripts: true
        }
    );


    const cssPath = vscode.Uri.file(context.asAbsolutePath(path.join('media','style.css')));
    const cssUri = panel.webview.asWebviewUri(cssPath);

    const htmlPath = context.asAbsolutePath(path.join('media','index.html'));
    const cssUriString = cssUri.toString();
    const htmlContent = fs.readFileSync(htmlPath, 'utf-8').replace(/{{style.css}}/g, cssUriString);
    panel.webview.html = htmlContent;

    // Listen to messages from the webview
    panel.webview.onDidReceiveMessage(async message => {
        // eslint-disable-next-line default-case, @typescript-eslint/no-unsafe-member-access
        switch (message.type) {
            case 'query':
                // Mockup: Replace this with actual API call
                // eslint-disable-next-line no-case-declarations, @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-member-access
                const apiResponse = await ApiCall(message.text);  
                void panel.webview.postMessage({
                    type: 'response',
                    text: apiResponse
                });
                break;
        }
    }, undefined);
}

// Mock API call: Replace with actual call to your API
// eslint-disable-next-line @typescript-eslint/require-await

const headers = {
    "Content-Type": "application/json",
    "Authorization": "",
  };


function getGeneratedText(json: any): string{
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return, @typescript-eslint/no-unsafe-member-access
    return json?.generated_text ?? json?.[0].generated_text ?? "";
}
async function ApiCall(query: string): Promise<string> {
    try {
        // eslint-disable-next-line @typescript-eslint/naming-convention
        const query_data = {
            inputs: query,
            parameters: {
                max_new_tokens: 64,
            }
            };
        // eslint-disable-next-line @typescript-eslint/no-unsafe-argument

        const config = workspace.getConfiguration("DevAssistant") as WorkspaceConfiguration & HFCodeConfig;
        const { chatEndpoint } = config;
        const response = await fetch(chatEndpoint, {
            method: "POST",
            headers,
            body: JSON.stringify(query_data),
          });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();  // Assuming your API returns JSON
        return getGeneratedText(data);  // Adjust this based on the structure of your API response

    } catch (error) {
        console.error('Error calling API', error);
        throw error;
    }
}

