package dev.penguinzdiscordintegration.minecraft;

import dev.penguinzdiscordintegration.config.BridgeConfig;
import dev.penguinzdiscordintegration.discord.DiscordBotManager;
import dev.penguinzdiscordintegration.linking.LinkManager;
import net.minecraft.ChatFormatting;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.damagesource.DamageSource;

public final class DeathListener {
    private final BridgeConfig config;
    private final DiscordBotManager discordBotManager;
    private final LinkManager linkManager;

    public DeathListener(BridgeConfig config, DiscordBotManager discordBotManager, LinkManager linkManager) {
        this.config = config;
        this.discordBotManager = discordBotManager;
        this.linkManager = linkManager;
    }

    public void onDeath(ServerPlayer player, DamageSource source) {
        if (!config.features.deathEmbeds || player == null || source == null) {
            return;
        }
        if (linkManager.isHiddenFromDiscord(player)) {
            return;
        }
        String deathMessage = source.getLocalizedDeathMessage(player).getString();
        String stripped = ChatFormatting.stripFormatting(deathMessage);
        discordBotManager.sendDeath(player, stripped == null ? deathMessage : stripped);
    }
}

