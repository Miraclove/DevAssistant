/* eslint-disable @typescript-eslint/no-unsafe-argument */
/* eslint-disable no-underscore-dangle */
/* eslint-disable no-param-reassign */
/* eslint-disable @typescript-eslint/no-unused-vars */
import * as vscode from "vscode";
import * as fs from 'fs'
import { WorkspaceConfiguration, workspace} from "vscode";
import type { Config as HFCodeConfig } from "./configTemplates"

// eslint-disable-next-line import/prefer-default-export
export class ChatSidebarViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "vscodeSidebar.openview";


  constructor(private readonly _extensionUri: vscode.Uri) {}

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext<unknown>,
    _token: vscode.CancellationToken
  ): void | Thenable<void> {

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._extensionUri],
    };

    const cssPath = vscode.Uri.joinPath(this._extensionUri, 'media', 'style.css')
    const cssUri = webviewView.webview.asWebviewUri(cssPath);

    const htmlPath = vscode.Uri.joinPath(this._extensionUri, 'media', 'index.html')
    const cssUriString = cssUri.toString();
    const config = workspace.getConfiguration("DevAssistant") as WorkspaceConfiguration & HFCodeConfig;
    const { chatEndpoint } = config;
    const htmlContent = fs.readFileSync(htmlPath.fsPath, 'utf-8').replace(/{{style.css}}/g, cssUriString).replace(/{{chatEndpoint}}/g, chatEndpoint);
    webviewView.webview.html = htmlContent;
  }
  
}

