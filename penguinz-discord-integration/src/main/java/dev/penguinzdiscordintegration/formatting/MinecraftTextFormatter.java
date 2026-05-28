package dev.penguinzdiscordintegration.formatting;

import net.minecraft.ChatFormatting;
import net.minecraft.network.chat.ClickEvent;
import net.minecraft.network.chat.Component;
import net.minecraft.network.chat.HoverEvent;
import net.minecraft.network.chat.MutableComponent;
import net.minecraft.network.chat.TextColor;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerPlayer;

import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

public final class MinecraftTextFormatter {
    public void broadcastDiscordMessage(MinecraftServer server, DiscordIncomingMessage message) {
        broadcastDiscordMessage(server, message, player -> true);
    }

    public void broadcastDiscordMessage(MinecraftServer server, DiscordIncomingMessage message, Predicate<ServerPlayer> recipientFilter) {
        List<Component> components = toComponents(message);
        if (components.isEmpty()) {
            return;
        }
        Predicate<ServerPlayer> filter = recipientFilter == null ? player -> true : recipientFilter;
        for (ServerPlayer player : server.getPlayerList().getPlayers()) {
            if (!filter.test(player)) {
                continue;
            }
            for (Component component : components) {
                player.sendSystemMessage(component);
            }
        }
    }

    private List<Component> toComponents(DiscordIncomingMessage message) {
        List<Component> components = new ArrayList<>();

        if (message.replyAuthor() != null && !message.replyAuthor().isBlank()) {
            components.add(replyLine(message));
        }

        if (message.content() != null && !message.content().isBlank()) {
            components.add(chatLine(message));
        }

        for (Attachment attachment : message.attachments()) {
            components.add(attachmentLine(message, attachment));
        }

        return components;
    }

    private Component chatLine(DiscordIncomingMessage message) {
        return prefix(null)
                .append(Component.literal(" "))
                .append(username(message.authorName(), message.roleColor()))
                .append(Component.literal(": ").withStyle(ChatFormatting.GRAY))
                .append(Component.literal(message.content()).withStyle(ChatFormatting.WHITE));
    }

    private Component replyLine(DiscordIncomingMessage message) {
        String channelName = message.channelName() == null || message.channelName().isBlank()
                ? null
                : "#" + message.channelName();
        return prefix(channelName)
                .append(Component.literal(" "))
                .append(username(message.authorName(), message.roleColor()))
                .append(Component.literal(" replied to ").withStyle(ChatFormatting.GRAY))
                .append(Component.literal(message.replyAuthor()).withStyle(ChatFormatting.GOLD))
                .append(Component.literal(": \"" + message.replyQuote() + "\"").withStyle(ChatFormatting.GRAY));
    }

    private Component attachmentLine(DiscordIncomingMessage message, Attachment attachment) {
        return prefix(null)
                .append(Component.literal(" "))
                .append(username(message.authorName(), message.roleColor()))
                .append(Component.literal(" sent attachment: ").withStyle(ChatFormatting.GRAY))
                .append(clickableAttachment(attachment));
    }

    private MutableComponent prefix(String channelName) {
        String text = channelName == null ? "[Discord]" : "[Discord/" + channelName + "]";
        return Component.literal(text).withStyle(ChatFormatting.DARK_PURPLE);
    }

    private MutableComponent username(String name, Integer roleColor) {
        MutableComponent component = Component.literal(name == null || name.isBlank() ? "Discord" : name);
        if (roleColor != null && roleColor != 0) {
            return component.withStyle(style -> style.withColor(TextColor.fromRgb(roleColor)));
        }
        return component.withStyle(ChatFormatting.AQUA);
    }

    private Component clickableAttachment(Attachment attachment) {
        MutableComponent label = Component.literal(attachment.fileName()).withStyle(style -> style
                .withColor(ChatFormatting.AQUA)
                .withUnderlined(true));
        try {
            URI uri = URI.create(attachment.url());
            return label.withStyle(style -> style
                    .withClickEvent(new ClickEvent.OpenUrl(uri))
                    .withHoverEvent(new HoverEvent.ShowText(Component.literal(attachment.url()))));
        } catch (IllegalArgumentException e) {
            return label.append(Component.literal(" " + attachment.url()).withStyle(ChatFormatting.GRAY));
        }
    }

    public record DiscordIncomingMessage(
            String channelName,
            String authorName,
            Integer roleColor,
            String content,
            List<Attachment> attachments,
            String replyAuthor,
            String replyQuote
    ) {
    }

    public record Attachment(String fileName, String url) {
    }
}

