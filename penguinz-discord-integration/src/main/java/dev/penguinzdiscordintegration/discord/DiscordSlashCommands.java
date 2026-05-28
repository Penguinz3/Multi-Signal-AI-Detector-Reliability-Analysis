package dev.penguinzdiscordintegration.discord;

import dev.penguinzdiscordintegration.PenguinzDiscordIntegrationMod;
import dev.penguinzdiscordintegration.config.BridgeConfig;
import dev.penguinzdiscordintegration.linking.LinkManager;
import dev.penguinzdiscordintegration.linking.PlayerLink;
import dev.penguinzdiscordintegration.minecraft.ServerStatusProvider;
import dev.penguinzdiscordintegration.minecraft.ServerStatusProvider.ServerSnapshot;
import net.dv8tion.jda.api.JDA;
import net.dv8tion.jda.api.entities.Guild;
import net.dv8tion.jda.api.entities.Member;
import net.dv8tion.jda.api.entities.Role;
import net.dv8tion.jda.api.events.guild.member.GuildMemberRemoveEvent;
import net.dv8tion.jda.api.events.interaction.command.SlashCommandInteractionEvent;
import net.dv8tion.jda.api.hooks.ListenerAdapter;
import net.dv8tion.jda.api.interactions.InteractionHook;
import net.dv8tion.jda.api.interactions.commands.Command;
import net.dv8tion.jda.api.interactions.commands.OptionMapping;
import net.dv8tion.jda.api.interactions.commands.OptionType;
import net.dv8tion.jda.api.interactions.commands.build.OptionData;
import net.dv8tion.jda.api.interactions.commands.build.SubcommandData;
import net.minecraft.ChatFormatting;
import net.minecraft.network.chat.Component;
import net.minecraft.server.level.ServerPlayer;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public final class DiscordSlashCommands extends ListenerAdapter {
    private final BridgeConfig config;
    private final ServerStatusProvider statusProvider;
    private final LinkManager linkManager;

    public DiscordSlashCommands(BridgeConfig config, ServerStatusProvider statusProvider, LinkManager linkManager) {
        this.config = config;
        this.statusProvider = statusProvider;
        this.linkManager = linkManager;
    }

    public void register(JDA jda) {
        OptionData settingKey = new OptionData(OptionType.STRING, "key", "Linked account setting key", false)
                .addChoices(settingChoices());
        OptionData settingValue = new OptionData(OptionType.BOOLEAN, "value", "New setting value", false);

        var command = net.dv8tion.jda.api.interactions.commands.build.Commands.slash(
                        "mc",
                        "Minecraft server bridge commands"
                )
                .addSubcommands(
                        new SubcommandData("status", "Show Minecraft server status"),
                        new SubcommandData("players", "Show online Minecraft players"),
                        new SubcommandData("link", "Link your Discord account to your Minecraft account")
                                .addOption(OptionType.INTEGER, "code", "Code from /discord link in Minecraft", true),
                        new SubcommandData("unlink", "Unlink your Discord account from Minecraft"),
                        new SubcommandData("settings", "View or update linked account settings")
                                .addOptions(settingKey, settingValue)
                );

        Guild guild = BridgeConfig.isBlank(config.guildId) ? null : jda.getGuildById(config.guildId);
        if (guild != null) {
            guild.updateCommands().addCommands(command).queue(
                    success -> PenguinzDiscordIntegrationMod.LOGGER.info("Registered Discord slash commands in guild {}", guild.getId()),
                    error -> PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to register guild slash commands", error)
            );
        } else {
            jda.updateCommands().addCommands(command).queue(
                    success -> PenguinzDiscordIntegrationMod.LOGGER.info("Registered global Discord slash commands"),
                    error -> PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to register global slash commands", error)
            );
        }
    }

    @Override
    public void onSlashCommandInteraction(SlashCommandInteractionEvent event) {
        if (!"mc".equals(event.getName())) {
            return;
        }

        String subcommand = event.getSubcommandName();
        event.deferReply(isPrivateSubcommand(subcommand)).queue(
                hook -> handleDeferred(event, hook),
                error -> PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to defer /mc slash command", error)
        );
    }

    @Override
    public void onGuildMemberRemove(GuildMemberRemoveEvent event) {
        if (config.accountLinking.unlinkOnDiscordLeave) {
            linkManager.unlinkDiscord(event.getUser().getId());
        }
    }

    private void handleDeferred(SlashCommandInteractionEvent event, InteractionHook hook) {
        String subcommand = event.getSubcommandName();
        switch (subcommand == null ? "" : subcommand) {
            case "link" -> handleLink(event, hook);
            case "unlink" -> handleUnlink(event, hook);
            case "settings" -> handleSettings(event, hook);
            default -> handleStatusCommand(event, hook);
        }
    }

    private void handleStatusCommand(SlashCommandInteractionEvent event, InteractionHook hook) {
        statusProvider.snapshotAsync()
                .orTimeout(3, TimeUnit.SECONDS)
                .whenComplete((snapshot, error) -> {
                    if (error != null) {
                        hook.sendMessage("Could not read Minecraft server status.").queue();
                        return;
                    }

                    String subcommand = event.getSubcommandName();
                    String response = "players".equals(subcommand) ? players(snapshot) : status(snapshot);
                    hook.sendMessage(response).queue();
                });
    }

    private void handleLink(SlashCommandInteractionEvent event, InteractionHook hook) {
        if (!linkManager.isEnabled()) {
            hook.sendMessage("Account linking is disabled on this server.").queue();
            return;
        }
        if (!memberHasRequiredRole(event)) {
            hook.sendMessage("You do not have a required Discord role to link your account.").queue();
            return;
        }

        OptionMapping codeOption = event.getOption("code");
        if (codeOption == null) {
            hook.sendMessage("Run `/discord link` in Minecraft first, then use the code here.").queue();
            return;
        }

        LinkManager.LinkAttempt attempt = linkManager.linkDiscordUser(event.getUser().getId(), codeOption.getAsInt());
        switch (attempt.status()) {
            case LINKED -> {
                hook.sendMessage("Your Discord account is now linked to Minecraft.").queue();
                syncLinkedMember(event, null);
                notifyLinkedPlayer(attempt.playerUuid(), event.getUser().getName(), event);
            }
            case DISABLED -> hook.sendMessage("Account linking is disabled on this server.").queue();
            case INVALID_CODE -> hook.sendMessage("That link code is invalid. Run `/discord link` in Minecraft for a fresh code.").queue();
            case CODE_EXPIRED -> hook.sendMessage("That link code expired. Run `/discord link` in Minecraft again.").queue();
            case DISCORD_ALREADY_LINKED -> hook.sendMessage("Your Discord account is already linked.").queue();
            case PLAYER_ALREADY_LINKED -> hook.sendMessage("That Minecraft account is already linked.").queue();
        }
    }

    private void handleUnlink(SlashCommandInteractionEvent event, InteractionHook hook) {
        PlayerLink link = linkManager.getByDiscordId(event.getUser().getId()).orElse(null);
        boolean unlinked = linkManager.unlinkDiscord(event.getUser().getId());
        if (unlinked && link != null) {
            removeSyncedRole(event);
        }
        hook.sendMessage(unlinked ? "Your Discord account has been unlinked." : "Your Discord account is not linked.").queue();
    }

    private void handleSettings(SlashCommandInteractionEvent event, InteractionHook hook) {
        PlayerLink link = linkManager.getByDiscordId(event.getUser().getId()).orElse(null);
        if (link == null) {
            hook.sendMessage("Your Discord account is not linked. Run `/discord link` in Minecraft first.").queue();
            return;
        }

        OptionMapping keyOption = event.getOption("key");
        OptionMapping valueOption = event.getOption("value");
        if (keyOption == null) {
            hook.sendMessage(linkManager.settingsSummary(link)).queue();
            return;
        }

        String key = keyOption.getAsString();
        if (valueOption == null) {
            hook.sendMessage(linkManager.settingValue(link, key)
                    .map(value -> key + ": " + value)
                    .orElse("Unknown setting: " + key)).queue();
            return;
        }

        LinkManager.SettingUpdate update = linkManager.updateSetting(event.getUser().getId(), key, valueOption.getAsBoolean());
        hook.sendMessage(settingUpdateMessage(update, key)).queue();
    }

    private void notifyLinkedPlayer(UUID playerUuid, String discordName, SlashCommandInteractionEvent event) {
        if (playerUuid == null) {
            return;
        }
        statusProvider.executeOnServerThread(server -> {
            ServerPlayer player = server.getPlayerList().getPlayer(playerUuid);
            if (player != null) {
                syncLinkedMember(event, player.getGameProfile().name());
                player.sendSystemMessage(Component.literal("Your Minecraft account is now linked with Discord user ")
                        .withStyle(ChatFormatting.GREEN)
                        .append(Component.literal(discordName).withStyle(ChatFormatting.AQUA)));
            }
        });
    }

    private void syncLinkedMember(SlashCommandInteractionEvent event, String playerName) {
        Guild guild = event.getGuild();
        if (guild == null) {
            return;
        }
        Member member = event.getMember();
        if (member != null) {
            applyLinkedAccountSync(guild, member, playerName);
            return;
        }
        guild.retrieveMemberById(event.getUser().getId()).queue(
                retrieved -> applyLinkedAccountSync(guild, retrieved, playerName),
                error -> PenguinzDiscordIntegrationMod.LOGGER.warn("Could not retrieve Discord member {} for account sync", event.getUser().getId(), error)
        );
    }

    private void applyLinkedAccountSync(Guild guild, Member member, String playerName) {
        if (config.accountNicknameSyncEnabled() && playerName != null && !playerName.isBlank()) {
            member.modifyNickname(playerName).queue(
                    success -> { },
                    error -> PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to sync Discord nickname for {}", member.getId(), error)
            );
        }

        if (!shouldSyncLinkedRole()) {
            return;
        }
        Role role = guild.getRoleById(config.linkedRoleId());
        if (role == null) {
            PenguinzDiscordIntegrationMod.LOGGER.warn("Configured linked Discord role {} was not found.", config.linkedRoleId());
            return;
        }
        if (!member.getRoles().contains(role)) {
            guild.addRoleToMember(member, role).queue(
                    success -> { },
                    error -> PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to sync linked Discord role for {}", member.getId(), error)
            );
        }
    }

    private void removeSyncedRole(SlashCommandInteractionEvent event) {
        if (!shouldSyncLinkedRole() || event.getGuild() == null) {
            return;
        }
        Guild guild = event.getGuild();
        Role role = guild.getRoleById(config.linkedRoleId());
        if (role == null) {
            return;
        }
        Member member = event.getMember();
        if (member != null) {
            removeRoleIfPresent(guild, member, role);
            return;
        }
        guild.retrieveMemberById(event.getUser().getId()).queue(
                retrieved -> removeRoleIfPresent(guild, retrieved, role),
                error -> { }
        );
    }

    private void removeRoleIfPresent(Guild guild, Member member, Role role) {
        if (member.getRoles().contains(role)) {
            guild.removeRoleFromMember(member, role).queue(
                    success -> { },
                    error -> PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to remove linked Discord role from {}", member.getId(), error)
            );
        }
    }

    private boolean shouldSyncLinkedRole() {
        return config.accountLinking.roleSync && !BridgeConfig.isBlank(config.linkedRoleId());
    }

    private String settingUpdateMessage(LinkManager.SettingUpdate update, String key) {
        return switch (update) {
            case UPDATED -> "Updated " + key + ".";
            case NOT_LINKED -> "Your Discord account is not linked.";
            case UNKNOWN_KEY -> "Unknown setting: " + key;
            case BLOCKED -> "That setting is locked by the server config.";
        };
    }

    private boolean memberHasRequiredRole(SlashCommandInteractionEvent event) {
        if (config.accountLinking.requiredRoles.length == 0 || event.getMember() == null) {
            return true;
        }
        Set<String> required = Arrays.stream(config.accountLinking.requiredRoles)
                .filter(roleId -> roleId != null && !roleId.isBlank() && !"0".equals(roleId))
                .collect(Collectors.toSet());
        if (required.isEmpty()) {
            return true;
        }
        return event.getMember().getRoles().stream().anyMatch(role -> required.contains(role.getId()));
    }

    private boolean isPrivateSubcommand(String subcommand) {
        return "link".equals(subcommand) || "unlink".equals(subcommand) || "settings".equals(subcommand);
    }

    private List<Command.Choice> settingChoices() {
        return linkManager.settingKeys().stream()
                .map(key -> new Command.Choice(key, key))
                .toList();
    }

    private String status(ServerSnapshot snapshot) {
        if (!snapshot.online()) {
            return "\uD83D\uDD34 Server Offline";
        }
        return "\uD83D\uDFE2 Server Online\n"
                + "Players: " + snapshot.playersOnline() + "/" + snapshot.maxPlayers() + "\n"
                + "Uptime: " + ServerStatusProvider.formatDuration(snapshot.uptime()) + "\n"
                + "TPS: " + String.format(Locale.US, "%.1f", snapshot.tps());
    }

    private String players(ServerSnapshot snapshot) {
        if (!snapshot.online() || snapshot.players().isEmpty()) {
            return "Online players:\nNone";
        }
        StringBuilder builder = new StringBuilder("Online players:");
        for (String player : snapshot.players()) {
            builder.append("\n- ").append(player);
        }
        return builder.toString();
    }
}
