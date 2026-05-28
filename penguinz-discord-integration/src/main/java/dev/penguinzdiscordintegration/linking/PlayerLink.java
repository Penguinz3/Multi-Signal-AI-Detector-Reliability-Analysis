package dev.penguinzdiscordintegration.linking;

import java.util.UUID;

public final class PlayerLink {
    public String discordID = "";
    public String mcPlayerUUID = "";
    public String floodgateUUID = "";
    public PlayerSettings settings = new PlayerSettings();

    public PlayerLink() {
    }

    public PlayerLink(String discordId, UUID playerUuid, PlayerSettings settings) {
        this.discordID = discordId == null ? "" : discordId;
        this.mcPlayerUUID = playerUuid == null ? "" : playerUuid.toString();
        this.floodgateUUID = "";
        this.settings = settings == null ? new PlayerSettings() : settings;
    }

    public UUID javaUuid() {
        if (mcPlayerUUID == null || mcPlayerUUID.isBlank()) {
            return null;
        }
        try {
            return UUID.fromString(mcPlayerUUID);
        } catch (IllegalArgumentException ignored) {
            return null;
        }
    }

    public boolean hasJavaUuid(UUID uuid) {
        return uuid != null && mcPlayerUUID != null && mcPlayerUUID.equals(uuid.toString());
    }
}
