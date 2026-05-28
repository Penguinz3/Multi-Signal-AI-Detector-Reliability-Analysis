package dev.penguinzdiscordintegration.minecraft;

import com.mojang.brigadier.arguments.BoolArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.context.CommandContext;
import dev.penguinzdiscordintegration.config.BridgeConfig;
import dev.penguinzdiscordintegration.discord.DiscordBotManager;
import dev.penguinzdiscordintegration.linking.LinkManager;
import dev.penguinzdiscordintegration.linking.PlayerLink;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.minecraft.ChatFormatting;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.network.chat.ClickEvent;
import net.minecraft.network.chat.Component;
import net.minecraft.network.chat.HoverEvent;
import net.minecraft.server.level.ServerPlayer;

import java.net.URI;

public final class DiscordCommand {
    private DiscordCommand() {
    }

    public static void register(BridgeConfig config, LinkManager linkManager, DiscordBotManager discordBotManager) {
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
                dispatcher.register(Commands.literal("discord")
                        .executes(context -> invite(context, config))
                        .then(Commands.literal("link")
                                .executes(context -> link(context, linkManager)))
                        .then(Commands.literal("unlink")
                                .executes(context -> unlink(context, linkManager, discordBotManager)))
                        .then(Commands.literal("settings")
                                .executes(context -> settings(context, linkManager))
                                .then(Commands.literal("get")
                                        .executes(context -> settings(context, linkManager))
                                        .then(Commands.argument("key", StringArgumentType.word())
                                                .executes(context -> getSetting(context, linkManager))))
                                .then(Commands.literal("set")
                                        .then(Commands.argument("key", StringArgumentType.word())
                                                .then(Commands.argument("value", BoolArgumentType.bool())
                                                        .executes(context -> setSetting(context, linkManager))))))));
    }

    private static int invite(CommandContext<CommandSourceStack> context, BridgeConfig config) {
        if (config.discordInviteUrl == null || config.discordInviteUrl.isBlank()) {
            send(context, Component.literal("Discord invite is not configured.").withStyle(ChatFormatting.RED));
            return 0;
        }
        URI inviteUri;
        try {
            inviteUri = URI.create(config.discordInviteUrl);
        } catch (IllegalArgumentException e) {
            send(context, Component.literal("Discord invite is not a valid URL.").withStyle(ChatFormatting.RED));
            return 0;
        }

        Component message = Component.literal("Join the Discord: ")
                .withStyle(ChatFormatting.AQUA)
                .append(Component.literal(config.discordInviteUrl).withStyle(style -> style
                        .withColor(ChatFormatting.GOLD)
                        .withUnderlined(true)
                        .withClickEvent(new ClickEvent.OpenUrl(inviteUri))
                        .withHoverEvent(new HoverEvent.ShowText(Component.literal(config.discordInviteUrl)))));
        send(context, message);
        return 1;
    }

    private static int link(CommandContext<CommandSourceStack> context, LinkManager linkManager) {
        ServerPlayer player;
        try {
            player = context.getSource().getPlayerOrException();
        } catch (Exception e) {
            send(context, Component.literal("Only players can link Discord accounts.").withStyle(ChatFormatting.RED));
            return 0;
        }
        if (!linkManager.isEnabled()) {
            player.sendSystemMessage(Component.literal("Discord account linking is disabled.").withStyle(ChatFormatting.RED));
            return 0;
        }
        if (linkManager.isPlayerLinked(player.getUUID())) {
            player.sendSystemMessage(Component.literal("Your Minecraft account is already linked.").withStyle(ChatFormatting.RED));
            return 0;
        }

        int code = linkManager.generateLinkCode(player.getUUID());
        player.sendSystemMessage(Component.literal("Use ")
                .withStyle(ChatFormatting.GOLD)
                .append(Component.literal("/mc link code:" + code).withStyle(ChatFormatting.AQUA))
                .append(Component.literal(" in Discord to link this account.").withStyle(ChatFormatting.GOLD))
                .withStyle(style -> style.withHoverEvent(new HoverEvent.ShowText(Component.literal("Link code: " + code)))));
        return 1;
    }

    private static int unlink(CommandContext<CommandSourceStack> context, LinkManager linkManager, DiscordBotManager discordBotManager) {
        ServerPlayer player;
        try {
            player = context.getSource().getPlayerOrException();
        } catch (Exception e) {
            send(context, Component.literal("Only players can unlink Discord accounts.").withStyle(ChatFormatting.RED));
            return 0;
        }
        PlayerLink link = linkManager.getByPlayer(player.getUUID()).orElse(null);
        boolean unlinked = linkManager.unlinkPlayer(player.getUUID());
        if (unlinked && link != null) {
            discordBotManager.removeSyncedLinkedRole(link.discordID);
        }
        player.sendSystemMessage(Component.literal(unlinked
                ? "Your Discord account has been unlinked."
                : "Your Minecraft account is not linked.").withStyle(unlinked ? ChatFormatting.GREEN : ChatFormatting.YELLOW));
        return unlinked ? 1 : 0;
    }

    private static int settings(CommandContext<CommandSourceStack> context, LinkManager linkManager) {
        ServerPlayer player;
        try {
            player = context.getSource().getPlayerOrException();
        } catch (Exception e) {
            send(context, Component.literal("Only players can view linked account settings.").withStyle(ChatFormatting.RED));
            return 0;
        }
        PlayerLink link = linkManager.getByPlayer(player.getUUID()).orElse(null);
        if (link == null) {
            player.sendSystemMessage(Component.literal("Your Minecraft account is not linked. Run /discord link first.").withStyle(ChatFormatting.YELLOW));
            return 0;
        }
        player.sendSystemMessage(Component.literal(linkManager.settingsSummary(link)).withStyle(ChatFormatting.AQUA));
        return 1;
    }

    private static int getSetting(CommandContext<CommandSourceStack> context, LinkManager linkManager) {
        ServerPlayer player;
        try {
            player = context.getSource().getPlayerOrException();
        } catch (Exception e) {
            send(context, Component.literal("Only players can view linked account settings.").withStyle(ChatFormatting.RED));
            return 0;
        }
        String key = StringArgumentType.getString(context, "key");
        PlayerLink link = linkManager.getByPlayer(player.getUUID()).orElse(null);
        if (link == null) {
            player.sendSystemMessage(Component.literal("Your Minecraft account is not linked.").withStyle(ChatFormatting.YELLOW));
            return 0;
        }
        player.sendSystemMessage(Component.literal(linkManager.settingValue(link, key)
                .map(value -> key + ": " + value)
                .orElse("Unknown setting: " + key)).withStyle(ChatFormatting.AQUA));
        return 1;
    }

    private static int setSetting(CommandContext<CommandSourceStack> context, LinkManager linkManager) {
        ServerPlayer player;
        try {
            player = context.getSource().getPlayerOrException();
        } catch (Exception e) {
            send(context, Component.literal("Only players can update linked account settings.").withStyle(ChatFormatting.RED));
            return 0;
        }
        String key = StringArgumentType.getString(context, "key");
        boolean value = BoolArgumentType.getBool(context, "value");
        LinkManager.SettingUpdate update = linkManager.updateSetting(player.getUUID(), key, value);
        player.sendSystemMessage(Component.literal(settingUpdateMessage(update, key)).withStyle(update == LinkManager.SettingUpdate.UPDATED
                ? ChatFormatting.GREEN
                : ChatFormatting.YELLOW));
        return update == LinkManager.SettingUpdate.UPDATED ? 1 : 0;
    }

    private static String settingUpdateMessage(LinkManager.SettingUpdate update, String key) {
        return switch (update) {
            case UPDATED -> "Updated " + key + ".";
            case NOT_LINKED -> "Your Minecraft account is not linked.";
            case UNKNOWN_KEY -> "Unknown setting: " + key;
            case BLOCKED -> "That setting is locked by the server config.";
        };
    }

    private static void send(CommandContext<CommandSourceStack> context, Component message) {
        context.getSource().sendSuccess(() -> message, false);
    }
}
