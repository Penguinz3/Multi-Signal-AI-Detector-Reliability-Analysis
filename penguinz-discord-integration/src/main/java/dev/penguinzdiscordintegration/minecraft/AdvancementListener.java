package dev.penguinzdiscordintegration.minecraft;

import dev.penguinzdiscordintegration.config.BridgeConfig;
import dev.penguinzdiscordintegration.discord.DiscordBotManager;
import dev.penguinzdiscordintegration.linking.LinkManager;
import net.minecraft.ChatFormatting;
import net.minecraft.advancements.Advancement;
import net.minecraft.advancements.AdvancementHolder;
import net.minecraft.server.level.ServerPlayer;

public final class AdvancementListener {
    private final BridgeConfig config;
    private final DiscordBotManager discordBotManager;
    private final LinkManager linkManager;

    public AdvancementListener(BridgeConfig config, DiscordBotManager discordBotManager, LinkManager linkManager) {
        this.config = config;
        this.discordBotManager = discordBotManager;
        this.linkManager = linkManager;
    }

    public void onAdvancement(ServerPlayer player, AdvancementHolder advancementHolder) {
        if (!config.features.advancementEmbeds || player == null || advancementHolder == null) {
            return;
        }
        if (linkManager.isHiddenFromDiscord(player)) {
            return;
        }

        Advancement advancement = advancementHolder.value();
        advancement.display().ifPresent(display -> {
            if (!display.shouldAnnounceChat()) {
                return;
            }
            String title = clean(display.getTitle().getString());
            String description = clean(display.getDescription().getString());
            discordBotManager.sendAdvancement(player, title, description);
        });
    }

    private String clean(String value) {
        String stripped = ChatFormatting.stripFormatting(value);
        return stripped == null ? value : stripped;
    }
}

