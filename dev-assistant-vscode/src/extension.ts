import * as vscode from "vscode";
import type { TemplateKey } from "./configTemplates";
import { templates } from "./configTemplates";
import { registerCommands } from "./commandsHandler";
import tabnineExtensionProperties from "./globals/tabnineExtensionProperties";
import {
  COMPLETION_IMPORTS,
  handleImports,
  HANDLE_IMPORTS,
  getSelectionHandler,
} from "./selectionHandler";
import { registerStatusBar, setDefaultStatus } from "./statusBar/statusBar";
import { setTabnineExtensionContext } from "./globals/tabnineExtensionContext";
import installAutocomplete from "./autocompleteInstaller";
import handlePluginInstalled from "./handlePluginInstalled";
import { ChatSidebarViewProvider } from "./chatSidebarViewProvider";
import sendFile from "./fileCollector";

import { WorkspaceConfiguration, workspace} from "vscode";
import type { Config as HFCodeConfig } from "./configTemplates"



export async function activate(
  context: vscode.ExtensionContext
): Promise<void> {
  void initStartup(context);
  handleSelection(context);
  handleConfigTemplateChange(context);

  registerStatusBar(context);
  registerSideBar(context);
  registerSaveAction(context);

  // Do not await on this function as we do not want VSCode to wait for it to finish
  // before considering TabNine ready to operate.
  void backgroundInit(context);

  if (context.extensionMode !== vscode.ExtensionMode.Test) {
    handlePluginInstalled(context);
  }

  return Promise.resolve();
}

function initStartup(context: vscode.ExtensionContext): void {
  setTabnineExtensionContext(context);
}


// eslint-disable-next-line @typescript-eslint/require-await
async function backgroundInit(context: vscode.ExtensionContext) {
  setDefaultStatus();
  void registerCommands(context);

  await installAutocomplete(context);
}


export async function deactivate(){
}


function handleSelection(context: vscode.ExtensionContext) {
  if (tabnineExtensionProperties.isTabNineAutoImportEnabled) {
    context.subscriptions.push(
      vscode.commands.registerTextEditorCommand(
        COMPLETION_IMPORTS,
        getSelectionHandler(context)
      ),
      vscode.commands.registerTextEditorCommand(HANDLE_IMPORTS, handleImports)
    );
  }
}

function handleConfigTemplateChange(context: vscode.ExtensionContext) {
  const listener = vscode.workspace.onDidChangeConfiguration(async event => {
    if (event.affectsConfiguration('DevAssistant.configTemplate')) {

        // change config
        const config = vscode.workspace.getConfiguration("DevAssistant");
        const configKey = config.get("configTemplate") as TemplateKey;
        const template = templates[configKey];
        if(template){
          const updatePromises = Object.entries(template).map(([key, val]) => config.update(key, val, vscode.ConfigurationTarget.Global));
          await Promise.all(updatePromises);
        }
        // change endpoint
        
    }
  });
  context.subscriptions.push(listener);
}

function registerSaveAction(context: vscode.ExtensionContext) {
  context.subscriptions.push(vscode.workspace.onDidSaveTextDocument(document => {
    // This code will be executed every time a document is saved
    // void vscode.window.showInformationMessage(`Document saved: ${document.fileName}`);
    // only save file when the file type is lua
    const config = workspace.getConfiguration("DevAssistant") as WorkspaceConfiguration & HFCodeConfig;
    const { allowCollectData,targetFileType } = config;
    if(allowCollectData){
      if(document.fileName.endsWith(targetFileType)){
        void sendFile(document);
      }
    }
}));
}


function registerSideBar(context: vscode.ExtensionContext){
    // Console diagnostic information (console.log) and errors (console.error)
    // Will only be executed once when your extension is activated
    console.log('Congratulations, your extension "vscode-extension-sidebar-html" is active!');

    const provider = new ChatSidebarViewProvider(context.extensionUri);

    context.subscriptions.push(
      vscode.window.registerWebviewViewProvider(
        ChatSidebarViewProvider.viewType,
        provider
      )
      );

    context.subscriptions.push(
      vscode.commands.registerCommand("vscodeSidebar.menu.view", () => {
        const message = "Chat history is cleared.";
        void vscode.window.showInformationMessage(message);
      })
    );

    // Command has been defined in the package.json file
    // Provide the implementation of the command with registerCommand
    // CommandId parameter must match the command field in package.json
    const openWebView = vscode.commands.registerCommand('vscodeSidebar.openview', () => {
      // Display a message box to the user
      void vscode.window.showInformationMessage('Command " Sidebar View [vscodeSidebar.openview] " called.');
    });

    context.subscriptions.push(openWebView);
}


