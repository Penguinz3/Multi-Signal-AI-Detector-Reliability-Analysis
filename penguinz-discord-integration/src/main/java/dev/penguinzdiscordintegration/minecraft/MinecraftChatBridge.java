package dev.penguinzdiscordintegration.minecraft;

import dev.penguinzdiscordintegration.config.BridgeConfig;
import dev.penguinzdiscordintegration.discord.DiscordBotManager;
import dev.penguinzdiscordintegration.formatting.SmartPingResolver;
import dev.penguinzdiscordintegration.linking.LinkManager;
import net.minecraft.network.chat.Component;
import net.minecraft.server.level.ServerPlayer;

public final class MinecraftChatBridge {
    private final BridgeConfig config;
    private final DiscordBotManager discordBotManager;
    private final SmartPingResolver smartPingResolver;
    private final LinkManager linkManager;

    public MinecraftChatBridge(BridgeConfig config, DiscordBotManager discordBotManager, LinkManager linkManager) {
        this.config = config;
        this.discordBotManager = discordBotManager;
        this.smartPingResolver = new SmartPingResolver(config);
        this.linkManager = linkManager;
    }

    public void onMinecraftChat(ServerPlayer player, Component message) {
        if (!config.features.chatBridge || player == null || message == null) {
            return;
        }
        if (linkManager.isHiddenFromDiscord(player)) {
            return;
        }
        SmartPingResolver.Result resolved = smartPingResolver.resolve(player.getUUID(), message.getString());
        discordBotManager.sendMinecraftChat(player, resolved.content(), resolved.allowedUserMentions());
    }
}

