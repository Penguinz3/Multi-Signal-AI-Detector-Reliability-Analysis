package dev.penguinzdiscordintegration.config;

import java.util.LinkedHashMap;
import java.util.Map;

public final class BridgeConfig {
    public String botToken = "TOKEN_HERE";
    public String guildId = "123456789012345678";
    public String discordInviteUrl = "https://discord.gg/example";
    public String avatarProviderUrl = "https://mc-heads.net/avatar/{uuid}";
    public boolean includeServerName = false;
    public String serverName = "Minecraft Server";

    public Channels channels = new Channels();
    public Features features = new Features();
    public WebhookMode webhookMode = new WebhookMode();
    public SmartPings smartPings = new SmartPings();
    public VoiceMessages voiceMessages = new VoiceMessages();
    public ChannelStatus channelStatus = new ChannelStatus();
    public AccountLinking accountLinking = new AccountLinking();

    public void sanitize() {
        if (channels == null) {
            channels = new Channels();
        }
        if (features == null) {
            features = new Features();
        }
        if (webhookMode == null) {
            webhookMode = new WebhookMode();
        }
        if (smartPings == null) {
            smartPings = new SmartPings();
        }
        if (smartPings.aliases == null) {
            smartPings.aliases = new LinkedHashMap<>();
        }
        if (voiceMessages == null) {
            voiceMessages = new VoiceMessages();
        }
        if (channelStatus == null) {
            channelStatus = new ChannelStatus();
        }
        if (accountLinking == null) {
            accountLinking = new AccountLinking();
        }
        if (accountLinking.requiredRoles == null) {
            accountLinking.requiredRoles = new String[0];
        }
        if (accountLinking.settingsBlacklist == null) {
            accountLinking.settingsBlacklist = new String[0];
        }
        if (accountLinking.personalSettingsDefaults == null) {
            accountLinking.personalSettingsDefaults = new AccountLinking.PersonalSettingsDefaults();
        }
        if (isBlank(accountLinking.linkedRoleId) && !isBlank(accountLinking.linkedRoleID)) {
            accountLinking.linkedRoleId = accountLinking.linkedRoleID;
        }
        if (isBlank(accountLinking.linkedRoleID) && !isBlank(accountLinking.linkedRoleId)) {
            accountLinking.linkedRoleID = accountLinking.linkedRoleId;
        }
    }

    public boolean hasUsableToken() {
        return botToken != null && !botToken.isBlank() && !"TOKEN_HERE".equals(botToken);
    }

    public String chatChannelId() {
        return channels.chat;
    }

    public String eventsChannelId() {
        return isBlank(channels.events) ? channels.chat : channels.events;
    }

    public String voiceMessagesChannelId() {
        if (!isBlank(voiceMessages.sendToChannelId)) {
            return voiceMessages.sendToChannelId;
        }
        return isBlank(channels.voiceMessages) ? channels.chat : channels.voiceMessages;
    }

    public String statusTopicChannelId() {
        if (!isBlank(channelStatus.channelId)) {
            return channelStatus.channelId;
        }
        return isBlank(channels.statusTopic) ? channels.chat : channels.statusTopic;
    }

    public static boolean isBlank(String value) {
        return value == null || value.isBlank() || "0".equals(value);
    }

    public boolean accountNicknameSyncEnabled() {
        return accountLinking != null && (accountLinking.nicknameSync || accountLinking.shouldNickname);
    }

    public String linkedRoleId() {
        if (accountLinking == null) {
            return "0";
        }
        return isBlank(accountLinking.linkedRoleId) ? accountLinking.linkedRoleID : accountLinking.linkedRoleId;
    }

    public static final class Channels {
        public String chat = "123";
        public String events = "456";
        public String voiceMessages = "123";
        public String statusTopic = "123";
    }

    public static final class Features {
        public boolean chatBridge = true;
        public boolean discordToMinecraft = true;
        public boolean webhookMode = true;
        public boolean advancementEmbeds = true;
        public boolean deathEmbeds = true;
        public boolean serverStartStopMessages = true;
        public boolean joinLeaveLogs = true;
        public boolean replyContext = true;
        public boolean smartPings = true;
        public boolean voiceMessagesIntegration = true;
        public boolean channelStatusTopic = true;
        public boolean accountLinking = true;
    }

    public static final class WebhookMode {
        public boolean enabled = true;
        public boolean usePlayerName = true;
        public boolean usePlayerHeadAvatar = true;
        public boolean fallbackToBotMessage = true;
    }

    public static final class SmartPings {
        public boolean enabled = true;
        public Map<String, String> aliases = new LinkedHashMap<>();
        public int maxMentionsPerMessage = 3;
        public int cooldownSeconds = 20;
        public boolean allowEveryoneHere = false;
        public boolean allowRolePings = false;

        public SmartPings() {
            aliases.put("alex", "123456789012345678");
        }
    }

    public static final class VoiceMessages {
        public boolean enabled = true;
        public String sendToChannelId = "123";
        public boolean showPlayedStatus = true;
        public int maxDurationSeconds = 30;
    }

    public static final class AccountLinking {
        public boolean enabled = true;
        public boolean importDiscordIntegrationJson = true;
        public boolean unlinkOnDiscordLeave = true;
        public boolean requestGuildMembersIntent = false;
        public boolean roleSync = true;
        public boolean nicknameSync = false;
        public boolean shouldNickname = false;
        public String linkedRoleId = "0";
        public String linkedRoleID = "0";
        public int linkCodeExpirationSeconds = 600;
        public String[] requiredRoles = new String[0];
        public String[] settingsBlacklist = new String[0];
        public PersonalSettingsDefaults personalSettingsDefaults = new PersonalSettingsDefaults();

        public static final class PersonalSettingsDefaults {
            public boolean useDiscordNameInChannel = true;
            public boolean ignoreReactions = false;
            public boolean pingSound = true;
        }
    }

    public static final class ChannelStatus {
        public boolean enabled = true;
        public String channelId = "123";
        public int updateIntervalSeconds = 120;
        public String format = "\uD83D\uDFE2 Online | {playersOnline} players online | Uptime: {uptime}";
        public String offlineFormat = "\uD83D\uDD34 Offline | Last seen with {playersOnline} players online";
    }
}

