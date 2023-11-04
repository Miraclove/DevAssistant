/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable @typescript-eslint/no-unsafe-argument */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { Position, Range, TextDocument, WorkspaceConfiguration, workspace, window} from "vscode";
import * as vscode from 'vscode';
import {URL} from "url";
import fetch from "node-fetch";
import type { Config as HFCodeConfig } from "./configTemplates"
import { PREFIX, SUFFIX } from "./configTemplates"
import { AutocompleteResult, ResultEntry } from "./binary/requests/requests";
import { CHAR_LIMIT, FULL_BRAND_REPRESENTATION } from "./globals/consts";
import languages from "./globals/languages";
import { setDefaultStatus, setLoadingStatus } from "./statusBar/statusBar";
import { logInput, logOutput,logString } from "./outputChannels";
// import { getTabnineExtensionContext } from "./globals/tabnineExtensionContext";

const config = workspace.getConfiguration("DevAssistant") as WorkspaceConfiguration & HFCodeConfig;
const { targetFileType } = config;
export type CompletionType = "normal" | "snippet";

let didShowTokenWarning = false;
const errorShownDate: Record<number, number> = {};

export default async function runCompletion(
  document: TextDocument,
  position: Position,
  timeout?: number,
  currentSuggestionText = ""
): Promise<AutocompleteResult | null | undefined> {
  // only run on lua files
  if(!document.fileName.endsWith(targetFileType)){
    return null;
  }
  setLoadingStatus(FULL_BRAND_REPRESENTATION);
  const offset = document.offsetAt(position);
  const beforeStartOffset = Math.max(0, offset - CHAR_LIMIT);
  const afterEndOffset = offset + CHAR_LIMIT;
  const beforeStart = document.positionAt(beforeStartOffset);
  const afterEnd = document.positionAt(afterEndOffset);
  const prefix =  document.getText(new Range(beforeStart, position)) + currentSuggestionText;
  const suffix = document.getText(new Range(position, afterEnd));

  const config = workspace.getConfiguration("DevAssistant") as WorkspaceConfiguration & HFCodeConfig;
  const { modelEndpoint, isFillMode, autoregressiveModeTemplate, fillModeTemplate, stopTokens, tokensToClear, temperature, maxNewTokens } = config;

  // const context = getTabnineExtensionContext();
  // const apiToken = await context?.secrets.get("apiToken");

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

  // use FIM (fill-in-middle) mode if suffix is available
  const inputs = (isFillMode && suffix.trim()) ? fillModeTemplate.replace(PREFIX, prefix).replace(SUFFIX, suffix) : autoregressiveModeTemplate.replace(PREFIX, prefix).replace(SUFFIX, suffix);
  const filename = document.fileName;
  const data = {
    inputs,
    parameters: {
      max_new_tokens: clipMaxNewTokens(maxNewTokens as number),
      temperature,
      do_sample: temperature > 0,
      top_p: 0.95,
      stop: stopTokens
    },
    filename,
    type: "autocomplete"
  };
  logString(JSON.stringify(data))
  logInput(inputs, data.parameters);

  const headers = {
    "Content-Type": "application/json",
    "Authorization": "",
  };

  const res = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify(data),
  });

  if(!res.ok){
    console.error("Error sending a request", res.status, res.statusText);
    const FIVE_MIN_MS = 300_000;
    const showError = !errorShownDate[res.status] || Date.now() - errorShownDate[res.status] > FIVE_MIN_MS;
    if(showError){
      errorShownDate[res.status] = Date.now();
      await window.showErrorMessage(`HF Code Error: code - ${res.status}; msg - ${res.statusText}`);
    }
    // if user hasn't supplied API Token yet, ask user to supply one
    logString(`modelEndpoint不可用, 请在设置里设置 modelEndpoint,你需要一个可用的服务器提供服务。`)
    if(!didShowTokenWarning){
      didShowTokenWarning = true;
      logString(`modelEndpoint不可用, 请在设置里设置 modelEndpoint,你需要一个可用的服务器提供服务。`)
      await window.showInformationMessage(`modelEndpoint不可用,请在vscode设置里设置`,
        "设置"
      ).then(clicked => {
        if (clicked) {
          void vscode.commands.executeCommand('workbench.action.openSettings', '@ext:DevAssistant');
        }
      });
    }
    // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-member-access
    setDefaultStatus();
    return null;
  }

  const generatedTextRaw = getGeneratedText(await res.json());

  let generatedText = generatedTextRaw;
  if(generatedText.slice(0, inputs.length) === inputs){
    generatedText = generatedText.slice(inputs.length);
  }
  const regexToClear = new RegExp([...stopTokens, ...tokensToClear].map(token => token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'), 'g');
  generatedText = generatedText.replace(regexToClear, "");

  const resultEntry: ResultEntry = {
    new_prefix: generatedText,
    old_suffix: "",
    new_suffix: ""
  }

  const result: AutocompleteResult = {
    results: [resultEntry],
    old_prefix: "",
    user_message: [],
    is_locked: false,
  }

  setDefaultStatus();
  logOutput(generatedTextRaw);
  return result;
}

function getGeneratedText(json: any): string{
  // eslint-disable-next-line @typescript-eslint/no-unsafe-return, @typescript-eslint/no-unsafe-member-access
  return json?.generated_text ?? json?.[0].generated_text ?? "";
}

export type KnownLanguageType = keyof typeof languages;

export function getLanguageFileExtension(
  languageId: string
): string | undefined {
  return languages[languageId as KnownLanguageType];
}

export function getFileNameWithExtension(document: TextDocument): string {
  const { languageId, fileName } = document;
  if (!document.isUntitled) {
    return fileName;
  }
  const extension = getLanguageFileExtension(languageId);
  if (extension) {
    return fileName.concat(extension);
  }
  return fileName;
}

function clipMaxNewTokens(maxNewTokens: number): number {
  const MIN = 50;
  const MAX = 500;
  return Math.min(Math.max(maxNewTokens, MIN), MAX);
}