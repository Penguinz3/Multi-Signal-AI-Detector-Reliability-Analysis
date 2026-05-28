package dev.penguinzdiscordintegration;

import dev.penguinzdiscordintegration.config.BridgeConfig;
import dev.penguinzdiscordintegration.config.ConfigManager;
import dev.penguinzdiscordintegration.discord.DiscordBotManager;
import dev.penguinzdiscordintegration.minecraft.AdvancementListener;
import dev.penguinzdiscordintegration.minecraft.DeathListener;
import dev.penguinzdiscordintegration.minecraft.DiscordCommand;
import dev.penguinzdiscordintegration.minecraft.JoinLeaveListener;
import dev.penguinzdiscordintegration.minecraft.MinecraftChatBridge;
import dev.penguinzdiscordintegration.minecraft.ServerLifecycleListener;
import dev.penguinzdiscordintegration.minecraft.ServerStatusProvider;
import dev.penguinzdiscordintegration.linking.LinkManager;
import dev.penguinzdiscordintegration.voice.VoiceMessagesBridge;
import net.fabricmc.api.DedicatedServerModInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class PenguinzDiscordIntegrationMod implements DedicatedServerModInitializer {
    public static final String MOD_ID = "penguinz-discord-integration";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    private static MinecraftChatBridge minecraftChatBridge;
    private static AdvancementListener advancementListener;
    private static DeathListener deathListener;
    private static LinkManager linkManager;

    @Override
    public void onInitializeServer() {
        BridgeConfig config = ConfigManager.load();
        ServerStatusProvider statusProvider = new ServerStatusProvider();
        linkManager = new LinkManager(config);
        linkManager.load();
        DiscordBotManager discordBotManager = new DiscordBotManager(config, statusProvider, linkManager);

        minecraftChatBridge = new MinecraftChatBridge(config, discordBotManager, linkManager);
        advancementListener = new AdvancementListener(config, discordBotManager, linkManager);
        deathListener = new DeathListener(config, discordBotManager, linkManager);

        ServerLifecycleListener.register(discordBotManager, statusProvider);
        JoinLeaveListener.register(config, discordBotManager, linkManager);
        DiscordCommand.register(config, linkManager, discordBotManager);
        new VoiceMessagesBridge(config, discordBotManager, linkManager).register();

        LOGGER.info("Penguinz Discord Integration initialized.");
    }

    public static MinecraftChatBridge minecraftChatBridge() {
        return minecraftChatBridge;
    }

    public static AdvancementListener advancementListener() {
        return advancementListener;
    }

    public static DeathListener deathListener() {
        return deathListener;
    }

    public static LinkManager linkManager() {
        return linkManager;
    }
}

