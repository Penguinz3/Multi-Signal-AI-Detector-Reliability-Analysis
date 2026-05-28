package dev.penguinzdiscordintegration.formatting;

import dev.penguinzdiscordintegration.config.BridgeConfig;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

public final class PlayerAvatarResolver {
    private final BridgeConfig config;

    public PlayerAvatarResolver(BridgeConfig config) {
        this.config = config;
    }

    public String avatarUrl(UUID uuid, String playerName) {
        String template = config.avatarProviderUrl;
        if (template == null || template.isBlank()) {
            template = "https://mc-heads.net/avatar/{uuid}";
        }
        String uuidString = uuid.toString();
        return template
                .replace("{uuid}", uuidString)
                .replace("{uuid_dashless}", uuidString.replace("-", ""))
                .replace("{name}", URLEncoder.encode(playerName, StandardCharsets.UTF_8));
    }
}

