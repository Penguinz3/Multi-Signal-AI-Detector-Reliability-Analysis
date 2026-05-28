package dev.penguinzdiscordintegration.voice;

import dev.penguinzdiscordintegration.PenguinzDiscordIntegrationMod;
import dev.penguinzdiscordintegration.config.BridgeConfig;
import dev.penguinzdiscordintegration.discord.DiscordBotManager;
import dev.penguinzdiscordintegration.linking.LinkManager;
import net.minecraft.server.level.ServerPlayer;

import java.io.IOException;
import java.util.List;

public final class VoiceMessagesBridge {
    private static final int VOICE_MESSAGES_FRAMES_PER_SECOND = 50;

    private final BridgeConfig config;
    private final DiscordBotManager discordBotManager;
    private final LinkManager linkManager;
    private final VoiceMessageEventAdapter adapter;

    public VoiceMessagesBridge(BridgeConfig config, DiscordBotManager discordBotManager, LinkManager linkManager) {
        this.config = config;
        this.discordBotManager = discordBotManager;
        this.linkManager = linkManager;
        this.adapter = new VoiceMessageEventAdapter();
    }

    public void register() {
        if (!config.features.voiceMessagesIntegration || !config.voiceMessages.enabled) {
            return;
        }
        boolean registered = adapter.register(this::onVoiceMessage);
        if (registered) {
            PenguinzDiscordIntegrationMod.LOGGER.info("VoiceMessages integration registered through its public API.");
        } else {
            PenguinzDiscordIntegrationMod.LOGGER.info("VoiceMessages is not installed; optional voice-message forwarding is inactive.");
        }
    }

    private void onVoiceMessage(Object sender, List<byte[]> opusFrames, String targetName) {
        if (!(sender instanceof ServerPlayer player) || opusFrames == null || opusFrames.isEmpty()) {
            return;
        }
        if (linkManager.isHiddenFromDiscord(player)) {
            return;
        }

        int durationMillis = opusFrames.size() * 1000 / VOICE_MESSAGES_FRAMES_PER_SECOND;
        if (config.voiceMessages.maxDurationSeconds > 0 && durationMillis > config.voiceMessages.maxDurationSeconds * 1000) {
            PenguinzDiscordIntegrationMod.LOGGER.warn(
                    "Skipping VoiceMessages Discord upload from {} because {}ms exceeds configured maxDurationSeconds.",
                    player.getGameProfile().name(),
                    durationMillis
            );
            return;
        }

        try {
            byte[] oggAudio = OggOpusWriter.write(opusFrames);
            discordBotManager.sendVoiceMessage(player.getGameProfile().name(), player.getUUID(), durationMillis, oggAudio);
        } catch (IOException e) {
            PenguinzDiscordIntegrationMod.LOGGER.warn("Could not convert VoiceMessages Opus frames to Ogg for Discord upload", e);
        }
    }
}

