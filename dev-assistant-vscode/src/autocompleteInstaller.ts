import { languages, Disposable, ExtensionContext } from "vscode";
import getSuggestionMode, {
  SuggestionsMode,
} from "./capabilities/getSuggestionMode";
import {
  Capability,
  isCapabilityEnabled,
  onDidRefreshCapabilities,
} from "./capabilities/capabilities";
import registerInlineHandlers from "./inlineSuggestions/registerHandlers";
import provideCompletionItems from "./provideCompletionItems";
import { COMPLETION_TRIGGERS } from "./globals/consts";
import {setDefaultStatus} from "./statusBar/statusBar";
let subscriptions: Disposable[] = [];

export default async function installAutocomplete(
  context: ExtensionContext
): Promise<void> {
  context.subscriptions.push({
    dispose: () => uninstallAutocomplete(),
  });

  let installOptions = InstallOptions.get();

  await reinstallAutocomplete(installOptions);

  context.subscriptions.push(
    onDidRefreshCapabilities(() => {
      const newInstallOptions = InstallOptions.get();

      if (!newInstallOptions.equals(installOptions)) {
        void reinstallAutocomplete(newInstallOptions);
        installOptions = newInstallOptions;
      }
    })
  );
  setDefaultStatus()
}

async function reinstallAutocomplete({
  inlineEnabled,
  snippetsEnabled,
  autocompleteEnabled,
}: InstallOptions) {
  uninstallAutocomplete();

  subscriptions.push(
    ...(await registerInlineHandlers(inlineEnabled, snippetsEnabled))
  );

  if (autocompleteEnabled) {
    subscriptions.push(
      languages.registerCompletionItemProvider(
        { pattern: "**" },
        // 'lua',
        {
          provideCompletionItems,
        },
        ...COMPLETION_TRIGGERS
      )
    );
  }
  setDefaultStatus()
}

class InstallOptions {
  inlineEnabled: boolean;

  snippetsEnabled: boolean;

  autocompleteEnabled: boolean;

  constructor(
    inlineEnabled: boolean,
    snippetsEnabled: boolean,
    autocompleteEnabled: boolean
  ) {
    this.inlineEnabled = inlineEnabled;
    this.snippetsEnabled = snippetsEnabled;
    this.autocompleteEnabled = autocompleteEnabled;
  }

  public equals(other: InstallOptions): boolean {
    return (
      this.autocompleteEnabled === other.autocompleteEnabled &&
      this.inlineEnabled === other.inlineEnabled &&
      this.snippetsEnabled === other.snippetsEnabled
    );
  }

  public static get() {
    return new InstallOptions(
      isInlineEnabled(),
      isSnippetSuggestionsEnabled(),
      isAutoCompleteEnabled()
    );
  }
}

function uninstallAutocomplete() {
  subscriptions.forEach((s) => {
    s.dispose();
  });
  subscriptions = [];
  setDefaultStatus()
}

function isInlineEnabled() {
  return getSuggestionMode() === SuggestionsMode.INLINE;
}

function isSnippetSuggestionsEnabled() {
  return isCapabilityEnabled(Capability.SNIPPET_SUGGESTIONS);
}

function isAutoCompleteEnabled() {
  return getSuggestionMode() === SuggestionsMode.AUTOCOMPLETE;
}


// auto complete switch function
export async function autocompleteSwitch(context: ExtensionContext) {
  if (subscriptions.length === 0) {
      await installAutocomplete(context);
  } else {
      context.subscriptions.push({
          dispose: () => uninstallAutocomplete(),
      });
      uninstallAutocomplete();
  }
}

// get auto complete status, true for on, false for off
export function getAutocompleteStatus(): boolean {
  return subscriptions.length !== 0;
}
