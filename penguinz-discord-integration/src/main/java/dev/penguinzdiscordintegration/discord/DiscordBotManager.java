package dev.penguinzdiscordintegration.discord;

import dev.penguinzdiscordintegration.PenguinzDiscordIntegrationMod;
import dev.penguinzdiscordintegration.config.BridgeConfig;
import dev.penguinzdiscordintegration.formatting.DiscordEmbedFactory;
import dev.penguinzdiscordintegration.formatting.PlayerAvatarResolver;
import dev.penguinzdiscordintegration.linking.LinkManager;
import dev.penguinzdiscordintegration.linking.PlayerLink;
import dev.penguinzdiscordintegration.minecraft.ServerStatusProvider;
import net.dv8tion.jda.api.JDA;
import net.dv8tion.jda.api.JDABuilder;
import net.dv8tion.jda.api.entities.Activity;
import net.dv8tion.jda.api.entities.Guild;
import net.dv8tion.jda.api.entities.Member;
import net.dv8tion.jda.api.entities.MessageEmbed;
import net.dv8tion.jda.api.entities.Role;
import net.dv8tion.jda.api.entities.User;
import net.dv8tion.jda.api.entities.channel.concrete.TextChannel;
import net.dv8tion.jda.api.events.session.ReadyEvent;
import net.dv8tion.jda.api.hooks.ListenerAdapter;
import net.dv8tion.jda.api.requests.GatewayIntent;
import net.dv8tion.jda.api.requests.restaction.MessageCreateAction;
import net.dv8tion.jda.api.utils.FileUpload;
import net.minecraft.server.level.ServerPlayer;

import java.time.Instant;
import java.util.Collection;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

public final class DiscordBotManager {
    private final BridgeConfig config;
    private final ServerStatusProvider statusProvider;
    private final DiscordEmbedFactory embedFactory;
    private final PlayerAvatarResolver avatarResolver;
    private final WebhookManager webhookManager;
    private final LinkManager linkManager;
    private final DiscordSlashCommands slashCommands;
    private final ChannelStatusUpdater channelStatusUpdater;
    private final AtomicReference<JDA> jda = new AtomicReference<>();
    private final Queue<Consumer<JDA>> pendingReadyActions = new ConcurrentLinkedQueue<>();
    private volatile boolean ready;
    private volatile boolean unavailable;

    public DiscordBotManager(BridgeConfig config, ServerStatusProvider statusProvider, LinkManager linkManager) {
        this.config = config;
        this.statusProvider = statusProvider;
        this.embedFactory = new DiscordEmbedFactory(config);
        this.avatarResolver = new PlayerAvatarResolver(config);
        this.webhookManager = new WebhookManager(config);
        this.linkManager = linkManager;
        this.slashCommands = new DiscordSlashCommands(config, statusProvider, linkManager);
        this.channelStatusUpdater = new ChannelStatusUpdater(config, statusProvider, this);
    }

    public boolean start() {
        if (!config.hasUsableToken()) {
            unavailable = true;
            PenguinzDiscordIntegrationMod.LOGGER.warn("Discord bridge disabled until config/penguinz-discord-integration.json has a real botToken.");
            return false;
        }
        if (jda.get() != null) {
            return true;
        }

        try {
            unavailable = false;
            EnumSet<GatewayIntent> intents = EnumSet.of(GatewayIntent.GUILD_MESSAGES, GatewayIntent.MESSAGE_CONTENT);
            if (config.features.accountLinking
                    && config.accountLinking.enabled
                    && config.accountLinking.unlinkOnDiscordLeave
                    && config.accountLinking.requestGuildMembersIntent) {
                intents.add(GatewayIntent.GUILD_MEMBERS);
            }
            JDA built = JDABuilder.createDefault(config.botToken, intents)
                    .setActivity(Activity.watching("Minecraft chat"))
                    .addEventListeners(
                            new ReadyListener(),
                            new DiscordMessageHandler(config, statusProvider, linkManager),
                            slashCommands
                    )
                    .build();
            jda.set(built);
            return true;
        } catch (Exception e) {
            unavailable = true;
            PenguinzDiscordIntegrationMod.LOGGER.error("Failed to start Discord bot", e);
            return false;
        }
    }

    public void startChannelStatus() {
        channelStatusUpdater.start();
    }

    public void stopChannelStatus() {
        channelStatusUpdater.stop();
    }

    public void updateOnlineStatusTopic() {
        channelStatusUpdater.updateOnline();
    }

    public void updateOfflineStatusTopic() {
        channelStatusUpdater.updateOffline();
    }

    public void shutdownSoon() {
        JDA current = jda.get();
        if (current == null) {
            return;
        }
        CompletableFuture.delayedExecutor(2, TimeUnit.SECONDS).execute(() -> {
            try {
                current.shutdown();
            } catch (Exception e) {
                PenguinzDiscordIntegrationMod.LOGGER.warn("Error while shutting down Discord bot", e);
            }
        });
    }

    public void sendServerStarted() {
        if (config.features.serverStartStopMessages) {
            sendEmbed(config.eventsChannelId(), embedFactory.serverStarted());
        }
    }

    public void sendServerStopped() {
        if (config.features.serverStartStopMessages) {
            sendEmbed(config.eventsChannelId(), embedFactory.serverStopped());
        }
    }

    public void sendJoin(ServerPlayer player) {
        if (config.features.joinLeaveLogs && !linkManager.isHiddenFromDiscord(player)) {
            sendPlain(config.eventsChannelId(), player.getGameProfile().name() + " joined");
        }
    }

    public void sendLeave(ServerPlayer player) {
        if (config.features.joinLeaveLogs && !linkManager.isHiddenFromDiscord(player)) {
            sendPlain(config.eventsChannelId(), player.getGameProfile().name() + " left");
        }
    }

    public void sendAdvancement(ServerPlayer player, String advancementTitle, String advancementDescription) {
        if (config.features.advancementEmbeds && !linkManager.isHiddenFromDiscord(player)) {
            sendEmbed(config.eventsChannelId(), embedFactory.advancement(player, advancementTitle, advancementDescription));
        }
    }

    public void sendDeath(ServerPlayer player, String deathMessage) {
        if (config.features.deathEmbeds && !linkManager.isHiddenFromDiscord(player)) {
            sendEmbed(config.eventsChannelId(), embedFactory.death(player, deathMessage));
        }
    }

    public void sendMinecraftChat(ServerPlayer player, String content, Set<String> allowedUserMentions) {
        if (!config.features.chatBridge) {
            return;
        }
        if (linkManager.isHiddenFromDiscord(player)) {
            return;
        }

        boolean webhookEnabled = config.features.webhookMode && config.webhookMode.enabled;
        if (webhookEnabled) {
            sendMinecraftChatWebhook(player, content, allowedUserMentions);
        } else {
            sendMinecraftChatBot(player, content, allowedUserMentions);
        }
    }

    public void sendMinecraftChatBot(ServerPlayer player, String content, Set<String> allowedUserMentions) {
        String playerName = player.getGameProfile().name();
        sendMessage(config.chatChannelId(), playerName + ": " + content, allowedUserMentions);
    }

    public void sendVoiceMessage(String playerName, java.util.UUID playerUuid, int durationMillis, byte[] oggAudio) {
        if (!config.features.voiceMessagesIntegration || !config.voiceMessages.enabled) {
            return;
        }
        String channelId = config.voiceMessagesChannelId();
        MessageEmbed embed = embedFactory.voiceMessage(
                playerName,
                playerUuid,
                durationMillis,
                config.voiceMessages.showPlayedStatus,
                Instant.now()
        );
        String fileName = "minecraft-voice-" + playerName.replaceAll("[^A-Za-z0-9_.-]", "_") + ".ogg";

        runWhenReady(discord -> {
            TextChannel channel = textChannel(discord, channelId);
            if (channel == null) {
                return;
            }
            channel.sendMessageEmbeds(embed)
                    .addFiles(FileUpload.fromData(oggAudio, fileName))
                    .setAllowedMentions(Collections.emptySet())
                    .queue(
                            message -> { },
                            error -> PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to send VoiceMessages upload to Discord", error)
                    );
        });
    }

    public void syncLinkedAccount(ServerPlayer player) {
        if (player == null || !linkManager.isEnabled()) {
            return;
        }
        PlayerLink link = linkManager.getByPlayer(player.getUUID()).orElse(null);
        if (link == null || link.discordID == null || link.discordID.isBlank()) {
            return;
        }
        String playerName = player.getGameProfile().name();
        runWhenReady(discord -> {
            Guild guild = guild(discord);
            if (guild == null) {
                return;
            }
            guild.retrieveMemberById(link.discordID).queue(
                    member -> applyLinkedAccountSync(guild, member, playerName),
                    error -> PenguinzDiscordIntegrationMod.LOGGER.warn("Could not retrieve linked Discord member {} for sync", link.discordID, error)
            );
        });
    }

    public void removeSyncedLinkedRole(String discordId) {
        if (discordId == null || discordId.isBlank() || !shouldSyncLinkedRole()) {
            return;
        }
        runWhenReady(discord -> {
            Guild guild = guild(discord);
            if (guild == null) {
                return;
            }
            Role role = guild.getRoleById(config.linkedRoleId());
            if (role == null) {
                return;
            }
            guild.retrieveMemberById(discordId).queue(
                    member -> {
                        if (member.getRoles().contains(role)) {
                            guild.removeRoleFromMember(member, role).queue(
                                    success -> { },
                                    error -> PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to remove linked Discord role from {}", discordId, error)
                            );
                        }
                    },
                    error -> { }
            );
        });
    }

    public void sendPlain(String channelId, String content) {
        sendMessage(channelId, content, Set.of());
    }

    public void sendEmbed(String channelId, MessageEmbed embed) {
        if (embed == null || BridgeConfig.isBlank(channelId)) {
            return;
        }
        runWhenReady(discord -> {
            TextChannel channel = textChannel(discord, channelId);
            if (channel == null) {
                return;
            }
            channel.sendMessageEmbeds(embed)
                    .setAllowedMentions(Collections.emptySet())
                    .queue(
                            message -> { },
                            error -> PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to send Discord embed to channel {}", channelId, error)
                    );
        });
    }

    public void updateChannelTopic(String channelId, String topic) {
        if (BridgeConfig.isBlank(channelId) || topic == null || topic.isBlank()) {
            return;
        }
        runWhenReady(discord -> {
            TextChannel channel = textChannel(discord, channelId);
            if (channel == null) {
                return;
            }
            String clampedTopic = topic.length() > 1024 ? topic.substring(0, 1024) : topic;
            channel.getManager().setTopic(clampedTopic).queue(
                    success -> { },
                    error -> PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to update Discord channel topic for {}", channelId, error)
            );
        });
    }

    void runWhenReady(Consumer<JDA> action) {
        JDA current = jda.get();
        if (ready && current != null) {
            action.accept(current);
            return;
        }
        if (current == null && (!config.hasUsableToken() || unavailable)) {
            return;
        }
        pendingReadyActions.add(action);
    }

    private void sendMinecraftChatWebhook(ServerPlayer player, String content, Set<String> allowedUserMentions) {
        runWhenReady(discord -> {
            TextChannel channel = textChannel(discord, config.chatChannelId());
            if (channel == null) {
                if (config.webhookMode.fallbackToBotMessage) {
                    sendMinecraftChatBot(player, content, allowedUserMentions);
                }
                return;
            }

            PlayerLink link = linkManager.getByPlayer(player.getUUID())
                    .filter(playerLink -> playerLink.settings != null && playerLink.settings.useDiscordNameInChannel)
                    .orElse(null);
            DiscordWebhookIdentity cachedIdentity = cachedDiscordIdentity(discord, channel, link);
            if (cachedIdentity != null) {
                sendWebhook(channel, player, cachedIdentity.name(), cachedIdentity.avatarUrl(), content, allowedUserMentions);
                return;
            }
            if (link != null && link.discordID != null && !link.discordID.isBlank()) {
                channel.getGuild().retrieveMemberById(link.discordID).queue(
                        member -> sendWebhook(channel, player, member.getEffectiveName(), member.getEffectiveAvatarUrl(), content, allowedUserMentions),
                        error -> sendWebhook(channel, player, null, null, content, allowedUserMentions)
                );
                return;
            }
            sendWebhook(channel, player, null, null, content, allowedUserMentions);
        });
    }

    private void sendWebhook(
            TextChannel channel,
            ServerPlayer player,
            String linkedName,
            String linkedAvatarUrl,
            String content,
            Set<String> allowedUserMentions
    ) {
        String playerName = config.webhookMode.usePlayerName
                ? (linkedName == null || linkedName.isBlank() ? player.getGameProfile().name() : linkedName)
                : "Minecraft";
        String avatarUrl = null;
        if (config.webhookMode.usePlayerHeadAvatar) {
            avatarUrl = linkedAvatarUrl == null || linkedAvatarUrl.isBlank()
                    ? avatarResolver.avatarUrl(player.getUUID(), player.getGameProfile().name())
                    : linkedAvatarUrl;
        }
        webhookManager.send(
                channel,
                playerName,
                avatarUrl,
                content,
                allowedUserMentions,
                () -> {
                    if (config.webhookMode.fallbackToBotMessage) {
                        sendMinecraftChatBot(player, content, allowedUserMentions);
                    }
                }
        );
    }

    private DiscordWebhookIdentity cachedDiscordIdentity(JDA discord, TextChannel channel, PlayerLink link) {
        if (link == null || link.discordID == null || link.discordID.isBlank()) {
            return null;
        }
        Member member = channel.getGuild().getMemberById(link.discordID);
        if (member != null) {
            return new DiscordWebhookIdentity(member.getEffectiveName(), member.getEffectiveAvatarUrl());
        }
        User user = discord.getUserById(link.discordID);
        if (user != null) {
            return new DiscordWebhookIdentity(user.getName(), user.getEffectiveAvatarUrl());
        }
        return null;
    }

    private void sendMessage(String channelId, String content, Set<String> allowedUserMentions) {
        if (content == null || content.isBlank() || BridgeConfig.isBlank(channelId)) {
            return;
        }
        runWhenReady(discord -> {
            TextChannel channel = textChannel(discord, channelId);
            if (channel == null) {
                return;
            }
            MessageCreateAction action = channel.sendMessage(content);
            applyAllowedMentions(action, allowedUserMentions);
            action.queue(
                    message -> { },
                    error -> PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to send Discord message to channel {}", channelId, error)
            );
        });
    }

    private void applyAllowedMentions(MessageCreateAction action, Collection<String> allowedUserMentions) {
        action.setAllowedMentions(Collections.emptySet());
        if (allowedUserMentions != null && !allowedUserMentions.isEmpty()) {
            action.mentionUsers(allowedUserMentions);
        }
    }

    private TextChannel textChannel(JDA discord, String channelId) {
        if (BridgeConfig.isBlank(channelId)) {
            return null;
        }
        TextChannel channel = discord.getTextChannelById(channelId);
        if (channel == null) {
            PenguinzDiscordIntegrationMod.LOGGER.warn("Configured Discord text channel {} was not found or is not visible to the bot.", channelId);
        }
        return channel;
    }

    private Guild guild(JDA discord) {
        if (!BridgeConfig.isBlank(config.guildId)) {
            Guild configured = discord.getGuildById(config.guildId);
            if (configured != null) {
                return configured;
            }
            PenguinzDiscordIntegrationMod.LOGGER.warn("Configured Discord guild {} was not found or is not visible to the bot.", config.guildId);
        }

        TextChannel chatChannel = textChannel(discord, config.chatChannelId());
        if (chatChannel != null) {
            return chatChannel.getGuild();
        }
        if (discord.getGuilds().size() == 1) {
            return discord.getGuilds().getFirst();
        }
        return null;
    }

    private void applyLinkedAccountSync(Guild guild, Member member, String playerName) {
        if (config.accountNicknameSyncEnabled() && playerName != null && !playerName.isBlank()) {
            member.modifyNickname(playerName).queue(
                    success -> { },
                    error -> PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to sync Discord nickname for {}", member.getId(), error)
            );
        }

        if (!shouldSyncLinkedRole()) {
            return;
        }
        Role role = guild.getRoleById(config.linkedRoleId());
        if (role == null) {
            PenguinzDiscordIntegrationMod.LOGGER.warn("Configured linked Discord role {} was not found.", config.linkedRoleId());
            return;
        }
        if (!member.getRoles().contains(role)) {
            guild.addRoleToMember(member, role).queue(
                    success -> { },
                    error -> PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to sync linked Discord role for {}", member.getId(), error)
            );
        }
    }

    private boolean shouldSyncLinkedRole() {
        return config.accountLinking.roleSync && !BridgeConfig.isBlank(config.linkedRoleId());
    }

    private final class ReadyListener extends ListenerAdapter {
        @Override
        public void onReady(ReadyEvent event) {
            ready = true;
            slashCommands.register(event.getJDA());
            PenguinzDiscordIntegrationMod.LOGGER.info("Discord bot is ready as {}", event.getJDA().getSelfUser().getName());

            Consumer<JDA> action;
            while ((action = pendingReadyActions.poll()) != null) {
                try {
                    action.accept(Objects.requireNonNull(jda.get()));
                } catch (Exception e) {
                    PenguinzDiscordIntegrationMod.LOGGER.warn("Deferred Discord action failed", e);
                }
            }
        }
    }

    private record DiscordWebhookIdentity(String name, String avatarUrl) {
    }
}

