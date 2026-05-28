package dev.penguinzdiscordintegration.formatting;

import dev.penguinzdiscordintegration.config.BridgeConfig;
import net.dv8tion.jda.api.EmbedBuilder;
import net.dv8tion.jda.api.entities.MessageEmbed;
import net.minecraft.server.level.ServerPlayer;

import java.awt.Color;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.Locale;
import java.util.UUID;

public final class DiscordEmbedFactory {
    private static final Color ADVANCEMENT_COLOR = new Color(0xF1C40F);
    private static final Color DEATH_COLOR = new Color(0x95A5A6);
    private static final Color ONLINE_COLOR = new Color(0x2ECC71);
    private static final Color OFFLINE_COLOR = new Color(0xE74C3C);
    private static final Color VOICE_COLOR = new Color(0x7289DA);
    private static final DateTimeFormatter TIME_FORMAT =
            DateTimeFormatter.ofPattern("h:mm a", Locale.US).withZone(ZoneId.systemDefault());

    private final BridgeConfig config;
    private final PlayerAvatarResolver avatarResolver;

    public DiscordEmbedFactory(BridgeConfig config) {
        this.config = config;
        this.avatarResolver = new PlayerAvatarResolver(config);
    }

    public MessageEmbed advancement(ServerPlayer player, String advancementTitle, String advancementDescription) {
        String playerName = player.getGameProfile().name();
        EmbedBuilder builder = base("ðŸ† Advancement Made", ADVANCEMENT_COLOR)
                .setDescription(playerName + " completed **" + advancementTitle + "**"
                        + (advancementDescription.isBlank() ? "" : "\n" + advancementDescription))
                .addField("Player", playerName, true)
                .addField("Advancement", advancementTitle, true)
                .setThumbnail(avatarResolver.avatarUrl(player.getUUID(), playerName));

        if (!advancementDescription.isBlank()) {
            builder.addField("Description", advancementDescription, false);
        }
        return builder.build();
    }

    public MessageEmbed death(ServerPlayer player, String deathMessage) {
        String playerName = player.getGameProfile().name();
        return base("â˜ ï¸ Player Died", DEATH_COLOR)
                .setDescription(deathMessage)
                .setThumbnail(avatarResolver.avatarUrl(player.getUUID(), playerName))
                .build();
    }

    public MessageEmbed serverStarted() {
        return base("ðŸŸ¢ Server Started", ONLINE_COLOR).build();
    }

    public MessageEmbed serverStopped() {
        return base("ðŸ”´ Server Stopped", OFFLINE_COLOR).build();
    }

    public MessageEmbed voiceMessage(String playerName, UUID playerUuid, int durationMillis, boolean showPlayedStatus, Instant sentAt) {
        EmbedBuilder builder = base("ðŸŽ™ Minecraft Voice Message", VOICE_COLOR)
                .addField("From", playerName, true)
                .addField("Duration", formatVoiceDuration(durationMillis), true)
                .addField("Sent", TIME_FORMAT.format(sentAt), true)
                .setThumbnail(avatarResolver.avatarUrl(playerUuid, playerName));

        if (showPlayedStatus) {
            builder.addField("Status", "Not played", true);
        }
        return builder.build();
    }

    private EmbedBuilder base(String title, Color color) {
        EmbedBuilder builder = new EmbedBuilder()
                .setTitle(title)
                .setColor(color)
                .setTimestamp(Instant.now());
        if (config.includeServerName && config.serverName != null && !config.serverName.isBlank()) {
            builder.setFooter(config.serverName);
        }
        return builder;
    }

    private static String formatVoiceDuration(int durationMillis) {
        int totalSeconds = Math.max(0, Math.round(durationMillis / 1000.0F));
        int minutes = totalSeconds / 60;
        int seconds = totalSeconds % 60;
        return String.format(Locale.US, "%d:%02d", minutes, seconds);
    }
}

