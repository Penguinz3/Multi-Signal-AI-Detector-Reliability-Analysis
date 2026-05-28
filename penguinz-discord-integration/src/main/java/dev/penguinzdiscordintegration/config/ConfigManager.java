package dev.penguinzdiscordintegration.config;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import dev.penguinzdiscordintegration.PenguinzDiscordIntegrationMod;
import net.fabricmc.loader.api.FabricLoader;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;

public final class ConfigManager {
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create();
    private static final String FILE_NAME = "penguinz-discord-integration.json";

    private ConfigManager() {
    }

    public static BridgeConfig load() {
        Path path = FabricLoader.getInstance().getConfigDir().resolve(FILE_NAME);
        if (!Files.exists(path)) {
            BridgeConfig defaults = new BridgeConfig();
            save(path, defaults);
            return defaults;
        }

        try (Reader reader = Files.newBufferedReader(path)) {
            BridgeConfig config = GSON.fromJson(reader, BridgeConfig.class);
            if (config == null) {
                config = new BridgeConfig();
            }
            config.sanitize();
            save(path, config);
            return config;
        } catch (Exception e) {
            PenguinzDiscordIntegrationMod.LOGGER.error("Failed to read {}. Using defaults for this run.", path, e);
            BridgeConfig defaults = new BridgeConfig();
            defaults.sanitize();
            return defaults;
        }
    }

    private static void save(Path path, BridgeConfig config) {
        try {
            Files.createDirectories(path.getParent());
            try (Writer writer = Files.newBufferedWriter(path)) {
                GSON.toJson(config, writer);
            }
        } catch (IOException e) {
            PenguinzDiscordIntegrationMod.LOGGER.warn("Could not write Discord bridge config {}", path, e);
        }
    }
}

