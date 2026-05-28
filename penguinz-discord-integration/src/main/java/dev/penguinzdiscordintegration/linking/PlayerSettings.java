package dev.penguinzdiscordintegration.linking;

import dev.penguinzdiscordintegration.config.BridgeConfig;

public final class PlayerSettings {
    public boolean useDiscordNameInChannel = true;
    public boolean ignoreDiscordChatIngame = false;
    public boolean ignoreReactions = false;
    public boolean pingSound = true;
    public boolean hideFromDiscord = false;

    public PlayerSettings() {
    }

    public PlayerSettings(BridgeConfig.AccountLinking.PersonalSettingsDefaults defaults) {
        if (defaults != null) {
            this.useDiscordNameInChannel = defaults.useDiscordNameInChannel;
            this.ignoreReactions = defaults.ignoreReactions;
            this.pingSound = defaults.pingSound;
        }
    }
}
