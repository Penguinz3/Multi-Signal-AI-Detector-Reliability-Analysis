package dev.penguinzdiscordintegration.discord;

import dev.penguinzdiscordintegration.PenguinzDiscordIntegrationMod;
import dev.penguinzdiscordintegration.config.BridgeConfig;
import dev.penguinzdiscordintegration.minecraft.ServerStatusProvider;
import dev.penguinzdiscordintegration.minecraft.ServerStatusProvider.ServerSnapshot;

import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public final class ChannelStatusUpdater {
    private final BridgeConfig config;
    private final ServerStatusProvider statusProvider;
    private final DiscordBotManager botManager;
    private ScheduledExecutorService scheduler;

    public ChannelStatusUpdater(BridgeConfig config, ServerStatusProvider statusProvider, DiscordBotManager botManager) {
        this.config = config;
        this.statusProvider = statusProvider;
        this.botManager = botManager;
    }

    public synchronized void start() {
        if (!config.features.channelStatusTopic || !config.channelStatus.enabled || scheduler != null) {
            return;
        }
        int interval = Math.max(60, config.channelStatus.updateIntervalSeconds);
        if (interval != config.channelStatus.updateIntervalSeconds) {
            PenguinzDiscordIntegrationMod.LOGGER.warn("channelStatus.updateIntervalSeconds was below 60; using 60 seconds to avoid Discord topic spam.");
        }
        scheduler = Executors.newSingleThreadScheduledExecutor(task -> {
            Thread thread = new Thread(task, "PenguinzDiscordIntegration-ChannelStatus");
            thread.setDaemon(true);
            return thread;
        });
        scheduler.scheduleAtFixedRate(this::updateOnline, 0L, interval, TimeUnit.SECONDS);
    }

    public synchronized void stop() {
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler = null;
        }
    }

    public void updateOnline() {
        if (!config.features.channelStatusTopic || !config.channelStatus.enabled) {
            return;
        }
        statusProvider.snapshotAsync().whenComplete((snapshot, error) -> {
            if (error != null) {
                PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to build Discord channel status topic", error);
                return;
            }
            String template = snapshot.online() ? config.channelStatus.format : config.channelStatus.offlineFormat;
            botManager.updateChannelTopic(config.statusTopicChannelId(), format(template, snapshot));
        });
    }

    public void updateOffline() {
        if (!config.features.channelStatusTopic || !config.channelStatus.enabled) {
            return;
        }
        statusProvider.snapshotAsync().whenComplete((snapshot, error) -> {
            if (error != null) {
                PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to build offline Discord channel status topic", error);
                return;
            }
            botManager.updateChannelTopic(config.statusTopicChannelId(), format(config.channelStatus.offlineFormat, snapshot));
        });
    }

    private String format(String template, ServerSnapshot snapshot) {
        String uptime = snapshot.online() ? ServerStatusProvider.formatDuration(snapshot.uptime()) : "0s";
        return template
                .replace("{playersOnline}", Integer.toString(snapshot.playersOnline()))
                .replace("{maxPlayers}", Integer.toString(snapshot.maxPlayers()))
                .replace("{uptime}", uptime)
                .replace("{tps}", String.format(Locale.US, "%.1f", snapshot.tps()))
                .replace("{status}", snapshot.online() ? "Online" : "Offline");
    }
}

