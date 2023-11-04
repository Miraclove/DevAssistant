/* eslint-disable import/prefer-default-export */
/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable @typescript-eslint/no-unsafe-argument */
import * as vscode from "vscode";
import {URL} from "url";
import fetch from "node-fetch";
import { WorkspaceConfiguration, window, workspace} from "vscode";
import type { Config as HFCodeConfig } from "./configTemplates"
import { logString } from "./outputChannels";

let didShowTokenWarning = false;
const headers = {
    "Content-Type": "application/json",
    "Authorization": "",
  };


export default async function sendFile(document: vscode.TextDocument){
    const config = workspace.getConfiguration("DevAssistant") as WorkspaceConfiguration & HFCodeConfig;
    const { modelEndpoint } = config;
    let endpoint = ""
    try{
        // eslint-disable-next-line no-new
        new URL(modelEndpoint);
        endpoint = modelEndpoint;
    }catch(e){
        // if user hasn't supplied API Token yet, ask user to supply one
        if(!didShowTokenWarning){
        didShowTokenWarning = true;
        logString(`modelEndpoint不可用, 请在设置里设置 modelEndpoint,你需要一个可用的服务器提供服务。`)
        void window.showInformationMessage(`modelEndpoint不可用,请在vscode设置里设置`,
            "设置"
        ).then(clicked => {
            if (clicked) {
            void vscode.commands.executeCommand('workbench.action.openSettings', '@ext:DevAssistant');
            }
        });
        }
    }
    try {
        // eslint-disable-next-line @typescript-eslint/naming-convention
        const filename = document.fileName;
        const data = {
            inputs: document.getText(),
            parameters: {
                max_new_tokens: 64,
            },
            filename,
            type: "save"
        };
        // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
        const response = await fetch(endpoint, {
            method: "POST",
            headers,
            body: JSON.stringify(data),
          });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
    } catch (error) {
        console.error('Error calling API', error);
        throw error;
    }
}
