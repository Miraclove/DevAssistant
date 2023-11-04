import { window, commands, ExtensionContext } from "vscode";
import { PROJECT_OPEN_GITHUB_COMMAND } from "./globals/consts";
import { getTabnineExtensionContext } from "./globals/tabnineExtensionContext";
// import openPanel from "./chatwebview"
// eslint-disable-next-line import/no-cycle
import {autocompleteSwitch} from "./autocompleteInstaller"

export const SET_API_TOKEN_COMMAND = "DevAssistant::setApiToken";
export const STATUS_BAR_COMMAND = "TabNine.statusBar";

export function registerCommands(
  context: ExtensionContext
): void {
  context.subscriptions.push(
    commands.registerCommand(SET_API_TOKEN_COMMAND, setApiToken)
  );
  context.subscriptions.push(
    commands.registerCommand(STATUS_BAR_COMMAND, handleStatusBar())
  );
  context.subscriptions.push(
    commands.registerCommand(PROJECT_OPEN_GITHUB_COMMAND, () => {
      void autocompleteSwitch(context);
      // void openPanel(context);
    }),
  );
}



function handleStatusBar() {
  return (): void => {
    void commands.executeCommand(PROJECT_OPEN_GITHUB_COMMAND);
  };
}

async function setApiToken () {
  const context = getTabnineExtensionContext();
  const input = await window.showInputBox({
      prompt: 'Please enter your API token (find yours at hf.co/settings/token):',
      placeHolder: 'Your token goes here ...'
  });
  if (input !== undefined) {
    await context?.secrets.store('apiToken', input);
    void window.showInformationMessage(`DevAssistant: API Token was successfully saved`);
  }
};