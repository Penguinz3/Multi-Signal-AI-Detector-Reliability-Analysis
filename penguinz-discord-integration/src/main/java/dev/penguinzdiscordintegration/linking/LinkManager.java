package dev.penguinzdiscordintegration.linking;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonParser;
import dev.penguinzdiscordintegration.PenguinzDiscordIntegrationMod;
import dev.penguinzdiscordintegration.config.BridgeConfig;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.server.level.ServerPlayer;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.SecureRandom;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public final class LinkManager {
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create();
    private static final String LINK_FILE_NAME = "LinkedPlayers.json";

    private final BridgeConfig config;
    private final Path linkFile;
    private final Path legacyDiscordIntegrationFile;
    private final SecureRandom random = new SecureRandom();
    private final Map<String, PlayerLink> byDiscordId = new ConcurrentHashMap<>();
    private final Map<UUID, PlayerLink> byPlayerUuid = new ConcurrentHashMap<>();
    private final Map<Integer, PendingLink> pendingLinks = new ConcurrentHashMap<>();

    public LinkManager(BridgeConfig config) {
        this.config = config;
        Path configDir = FabricLoader.getInstance().getConfigDir()
                .resolve(PenguinzDiscordIntegrationMod.MOD_ID);
        this.linkFile = configDir.resolve(LINK_FILE_NAME);
        this.legacyDiscordIntegrationFile = FabricLoader.getInstance().getGameDir()
                .resolve("DiscordIntegration-Data")
                .resolve(LINK_FILE_NAME);
    }

    public void load() {
        if (!isEnabled()) {
            return;
        }

        try {
            Files.createDirectories(linkFile.getParent());
            if (Files.exists(linkFile)) {
                loadFrom(linkFile);
            } else if (config.accountLinking.importDiscordIntegrationJson && Files.exists(legacyDiscordIntegrationFile)) {
                loadFrom(legacyDiscordIntegrationFile);
                save();
                PenguinzDiscordIntegrationMod.LOGGER.info(
                        "Imported DiscordIntegration account links from {} into Penguinz Discord Integration.",
                        legacyDiscordIntegrationFile
                );
            } else {
                save();
            }
        } catch (Exception e) {
            PenguinzDiscordIntegrationMod.LOGGER.warn("Could not load Discord account links from {}", linkFile, e);
        }
    }

    public boolean isEnabled() {
        return config.features.accountLinking && config.accountLinking.enabled;
    }

    public Optional<PlayerLink> getByPlayer(UUID playerUuid) {
        if (!isEnabled() || playerUuid == null) {
            return Optional.empty();
        }
        return Optional.ofNullable(byPlayerUuid.get(playerUuid));
    }

    public Optional<PlayerLink> getByDiscordId(String discordId) {
        if (!isEnabled() || discordId == null || discordId.isBlank()) {
            return Optional.empty();
        }
        return Optional.ofNullable(byDiscordId.get(discordId));
    }

    public boolean isPlayerLinked(UUID playerUuid) {
        return getByPlayer(playerUuid).isPresent();
    }

    public boolean isDiscordLinked(String discordId) {
        return getByDiscordId(discordId).isPresent();
    }

    public boolean isHiddenFromDiscord(ServerPlayer player) {
        return player != null && getByPlayer(player.getUUID())
                .map(link -> link.settings != null && link.settings.hideFromDiscord)
                .orElse(false);
    }

    public boolean shouldReceiveDiscordChat(ServerPlayer player) {
        return player == null || getByPlayer(player.getUUID())
                .map(link -> link.settings == null || !link.settings.ignoreDiscordChatIngame)
                .orElse(true);
    }

    public int generateLinkCode(UUID playerUuid) {
        cleanupExpired();
        for (Map.Entry<Integer, PendingLink> entry : pendingLinks.entrySet()) {
            if (entry.getValue().playerUuid().equals(playerUuid)) {
                return entry.getKey();
            }
        }

        int code;
        do {
            code = 10_000 + random.nextInt(90_000);
        } while (pendingLinks.containsKey(code));
        pendingLinks.put(code, new PendingLink(Instant.now(), playerUuid));
        return code;
    }

    public LinkAttempt linkDiscordUser(String discordId, int code) {
        if (!isEnabled()) {
            return new LinkAttempt(LinkStatus.DISABLED, null, null);
        }
        cleanupExpired();
        PendingLink pending = pendingLinks.remove(code);
        if (pending == null) {
            return new LinkAttempt(LinkStatus.INVALID_CODE, null, null);
        }
        if (isExpired(pending)) {
            return new LinkAttempt(LinkStatus.CODE_EXPIRED, null, pending.playerUuid());
        }
        if (isDiscordLinked(discordId)) {
            return new LinkAttempt(LinkStatus.DISCORD_ALREADY_LINKED, getByDiscordId(discordId).orElse(null), pending.playerUuid());
        }
        if (isPlayerLinked(pending.playerUuid())) {
            return new LinkAttempt(LinkStatus.PLAYER_ALREADY_LINKED, getByPlayer(pending.playerUuid()).orElse(null), pending.playerUuid());
        }

        PlayerLink link = new PlayerLink(
                discordId,
                pending.playerUuid(),
                new PlayerSettings(config.accountLinking.personalSettingsDefaults)
        );
        addLink(link);
        save();
        return new LinkAttempt(LinkStatus.LINKED, link, pending.playerUuid());
    }

    public boolean unlinkDiscord(String discordId) {
        PlayerLink link = getByDiscordId(discordId).orElse(null);
        if (link == null) {
            return false;
        }
        removeLink(link);
        save();
        return true;
    }

    public boolean unlinkPlayer(UUID playerUuid) {
        PlayerLink link = getByPlayer(playerUuid).orElse(null);
        if (link == null) {
            return false;
        }
        removeLink(link);
        save();
        return true;
    }

    public SettingUpdate updateSetting(String discordId, String key, boolean value) {
        PlayerLink link = getByDiscordId(discordId).orElse(null);
        if (link == null) {
            return SettingUpdate.NOT_LINKED;
        }
        return updateSetting(link, key, value);
    }

    public SettingUpdate updateSetting(UUID playerUuid, String key, boolean value) {
        PlayerLink link = getByPlayer(playerUuid).orElse(null);
        if (link == null) {
            return SettingUpdate.NOT_LINKED;
        }
        return updateSetting(link, key, value);
    }

    public Optional<Boolean> settingValue(PlayerLink link, String key) {
        if (link == null || link.settings == null) {
            return Optional.empty();
        }
        return switch (normalKey(key)) {
            case "usediscordnameinchannel" -> Optional.of(link.settings.useDiscordNameInChannel);
            case "ignorediscordchatingame" -> Optional.of(link.settings.ignoreDiscordChatIngame);
            case "ignorereactions" -> Optional.of(link.settings.ignoreReactions);
            case "pingsound" -> Optional.of(link.settings.pingSound);
            case "hidefromdiscord" -> Optional.of(link.settings.hideFromDiscord);
            default -> Optional.empty();
        };
    }

    public String settingsSummary(PlayerLink link) {
        if (link == null || link.settings == null) {
            return "No linked settings found.";
        }
        StringBuilder builder = new StringBuilder("Linked account settings:");
        for (String key : settingKeys()) {
            builder.append("\n")
                    .append(key)
                    .append(": ")
                    .append(settingValue(link, key).orElse(false));
        }
        return builder.toString();
    }

    public List<String> settingKeys() {
        return List.of(
                "useDiscordNameInChannel",
                "ignoreDiscordChatIngame",
                "ignoreReactions",
                "pingSound",
                "hideFromDiscord"
        );
    }

    public Map<String, String> settingDescriptions() {
        Map<String, String> descriptions = new LinkedHashMap<>();
        descriptions.put("useDiscordNameInChannel", "Use your Discord name/avatar for webhook-mode Minecraft chat.");
        descriptions.put("ignoreDiscordChatIngame", "Hide Discord chat messages from you in Minecraft.");
        descriptions.put("ignoreReactions", "Reserved for reaction bridging compatibility.");
        descriptions.put("pingSound", "Reserved for ping sound compatibility.");
        descriptions.put("hideFromDiscord", "Hide your Minecraft chat/events from Discord.");
        return descriptions;
    }

    private SettingUpdate updateSetting(PlayerLink link, String key, boolean value) {
        if (isBlacklisted(key)) {
            return SettingUpdate.BLOCKED;
        }
        if (link.settings == null) {
            link.settings = new PlayerSettings(config.accountLinking.personalSettingsDefaults);
        }
        switch (normalKey(key)) {
            case "usediscordnameinchannel" -> link.settings.useDiscordNameInChannel = value;
            case "ignorediscordchatingame" -> link.settings.ignoreDiscordChatIngame = value;
            case "ignorereactions" -> link.settings.ignoreReactions = value;
            case "pingsound" -> link.settings.pingSound = value;
            case "hidefromdiscord" -> link.settings.hideFromDiscord = value;
            default -> {
                return SettingUpdate.UNKNOWN_KEY;
            }
        }
        addLink(link);
        save();
        return SettingUpdate.UPDATED;
    }

    private void loadFrom(Path path) throws IOException {
        byDiscordId.clear();
        byPlayerUuid.clear();
        if (!Files.exists(path) || Files.size(path) == 0L) {
            return;
        }
        try (Reader reader = Files.newBufferedReader(path)) {
            JsonArray json = JsonParser.parseReader(reader).getAsJsonArray();
            PlayerLink[] links = GSON.fromJson(json, PlayerLink[].class);
            if (links != null) {
                for (PlayerLink link : links) {
                    addLink(sanitized(link));
                }
            }
        }
    }

    private void addLink(PlayerLink link) {
        if (link == null || link.discordID == null || link.discordID.isBlank()) {
            return;
        }
        removeLink(link);
        byDiscordId.put(link.discordID, link);
        UUID uuid = link.javaUuid();
        if (uuid != null) {
            byPlayerUuid.put(uuid, link);
        }
    }

    private void removeLink(PlayerLink link) {
        if (link == null) {
            return;
        }
        if (link.discordID != null) {
            byDiscordId.remove(link.discordID);
        }
        UUID uuid = link.javaUuid();
        if (uuid != null) {
            byPlayerUuid.remove(uuid);
        }
        Collection<PlayerLink> existing = new ArrayList<>(byDiscordId.values());
        for (PlayerLink old : existing) {
            boolean sameDiscord = old.discordID != null && old.discordID.equals(link.discordID);
            boolean sameUuid = uuid != null && old.hasJavaUuid(uuid);
            if (sameDiscord || sameUuid) {
                if (old.discordID != null) {
                    byDiscordId.remove(old.discordID);
                }
                UUID oldUuid = old.javaUuid();
                if (oldUuid != null) {
                    byPlayerUuid.remove(oldUuid);
                }
            }
        }
    }

    private void save() {
        try {
            Files.createDirectories(linkFile.getParent());
            try (Writer writer = Files.newBufferedWriter(linkFile)) {
                GSON.toJson(new ArrayList<>(byDiscordId.values()), writer);
            }
        } catch (IOException e) {
            PenguinzDiscordIntegrationMod.LOGGER.warn("Could not save Discord account links to {}", linkFile, e);
        }
    }

    private PlayerLink sanitized(PlayerLink link) {
        if (link == null) {
            return null;
        }
        if (link.discordID == null) {
            link.discordID = "";
        }
        if (link.mcPlayerUUID == null) {
            link.mcPlayerUUID = "";
        }
        if (link.floodgateUUID == null) {
            link.floodgateUUID = "";
        }
        if (link.settings == null) {
            link.settings = new PlayerSettings(config.accountLinking.personalSettingsDefaults);
        }
        return link;
    }

    private void cleanupExpired() {
        pendingLinks.entrySet().removeIf(entry -> isExpired(entry.getValue()));
    }

    private boolean isExpired(PendingLink link) {
        int seconds = config.accountLinking.linkCodeExpirationSeconds;
        return seconds > 0 && Duration.between(link.createdAt(), Instant.now()).getSeconds() > seconds;
    }

    private boolean isBlacklisted(String key) {
        Set<String> blacklist = Arrays.stream(config.accountLinking.settingsBlacklist)
                .map(LinkManager::normalKey)
                .collect(Collectors.toUnmodifiableSet());
        return blacklist.contains(normalKey(key));
    }

    private static String normalKey(String key) {
        return key == null ? "" : key.toLowerCase(Locale.ROOT).replace("_", "");
    }

    private record PendingLink(Instant createdAt, UUID playerUuid) {
    }

    public enum LinkStatus {
        LINKED,
        INVALID_CODE,
        CODE_EXPIRED,
        DISCORD_ALREADY_LINKED,
        PLAYER_ALREADY_LINKED,
        DISABLED
    }

    public enum SettingUpdate {
        UPDATED,
        NOT_LINKED,
        UNKNOWN_KEY,
        BLOCKED
    }

    public record LinkAttempt(LinkStatus status, PlayerLink link, UUID playerUuid) {
    }
}
