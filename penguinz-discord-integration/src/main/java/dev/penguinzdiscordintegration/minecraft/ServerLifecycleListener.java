package dev.penguinzdiscordintegration.minecraft;

import dev.penguinzdiscordintegration.discord.DiscordBotManager;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;

public final class ServerLifecycleListener {
    private ServerLifecycleListener() {
    }

    public static void register(DiscordBotManager discordBotManager, ServerStatusProvider statusProvider) {
        ServerLifecycleEvents.SERVER_STARTED.register(server -> {
            statusProvider.onServerStarted(server);
            if (discordBotManager.start()) {
                discordBotManager.sendServerStarted();
                discordBotManager.startChannelStatus();
                discordBotManager.updateOnlineStatusTopic();
            }
        });

        ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
            discordBotManager.sendServerStopped();
            discordBotManager.updateOfflineStatusTopic();
            discordBotManager.stopChannelStatus();
        });

        ServerLifecycleEvents.SERVER_STOPPED.register(server -> {
            statusProvider.onServerStopped();
            discordBotManager.shutdownSoon();
        });

        ServerTickEvents.END_SERVER_TICK.register(statusProvider::recordTick);
    }
}

