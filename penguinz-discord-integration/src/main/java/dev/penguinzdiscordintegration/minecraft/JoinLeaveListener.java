package dev.penguinzdiscordintegration.minecraft;

import dev.penguinzdiscordintegration.config.BridgeConfig;
import dev.penguinzdiscordintegration.discord.DiscordBotManager;
import dev.penguinzdiscordintegration.linking.LinkManager;
import net.fabricmc.fabric.api.networking.v1.ServerPlayConnectionEvents;

public final class JoinLeaveListener {
    private JoinLeaveListener() {
    }

    public static void register(BridgeConfig config, DiscordBotManager discordBotManager, LinkManager linkManager) {
        ServerPlayConnectionEvents.JOIN.register((handler, sender, server) -> {
            discordBotManager.syncLinkedAccount(handler.player);
            if (config.features.joinLeaveLogs && !linkManager.isHiddenFromDiscord(handler.player)) {
                discordBotManager.sendJoin(handler.player);
            }
        });
        ServerPlayConnectionEvents.DISCONNECT.register((handler, server) -> {
            if (config.features.joinLeaveLogs && !linkManager.isHiddenFromDiscord(handler.player)) {
                discordBotManager.sendLeave(handler.player);
            }
        });
    }
}

