package dev.penguinzdiscordintegration.discord;

import dev.penguinzdiscordintegration.config.BridgeConfig;
import dev.penguinzdiscordintegration.formatting.MinecraftTextFormatter;
import dev.penguinzdiscordintegration.formatting.ReplyContextFormatter;
import dev.penguinzdiscordintegration.linking.LinkManager;
import dev.penguinzdiscordintegration.minecraft.ServerStatusProvider;
import net.dv8tion.jda.api.entities.Member;
import net.dv8tion.jda.api.entities.Message;
import net.dv8tion.jda.api.entities.MessageEmbed;
import net.dv8tion.jda.api.events.message.MessageReceivedEvent;
import net.dv8tion.jda.api.hooks.ListenerAdapter;

import java.util.ArrayList;
import java.util.List;

public final class DiscordMessageHandler extends ListenerAdapter {
    private final BridgeConfig config;
    private final ServerStatusProvider statusProvider;
    private final LinkManager linkManager;
    private final MinecraftTextFormatter minecraftTextFormatter = new MinecraftTextFormatter();

    public DiscordMessageHandler(BridgeConfig config, ServerStatusProvider statusProvider, LinkManager linkManager) {
        this.config = config;
        this.statusProvider = statusProvider;
        this.linkManager = linkManager;
    }

    @Override
    public void onMessageReceived(MessageReceivedEvent event) {
        if (!config.features.discordToMinecraft || !event.isFromGuild()) {
            return;
        }
        if (!event.getChannel().getId().equals(config.chatChannelId())) {
            return;
        }
        Message message = event.getMessage();
        if (message.isWebhookMessage() || event.getAuthor().isBot() || event.getAuthor().equals(event.getJDA().getSelfUser())) {
            return;
        }

        String content = ReplyContextFormatter.clean(message.getContentDisplay());
        if (content.isBlank()) {
            content = readableEmbedText(message.getEmbeds());
        }

        List<MinecraftTextFormatter.Attachment> attachments = message.getAttachments().stream()
                .map(attachment -> new MinecraftTextFormatter.Attachment(attachment.getFileName(), attachment.getUrl()))
                .toList();

        ReplyContext reply = replyContext(message);
        MinecraftTextFormatter.DiscordIncomingMessage incoming = new MinecraftTextFormatter.DiscordIncomingMessage(
                event.getChannel().getName(),
                authorName(message),
                roleColor(event.getMember()),
                content,
                attachments,
                reply.authorName(),
                reply.quote()
        );

        if ((content == null || content.isBlank()) && attachments.isEmpty()) {
            return;
        }

        statusProvider.executeOnServerThread(server ->
                minecraftTextFormatter.broadcastDiscordMessage(server, incoming, linkManager::shouldReceiveDiscordChat)
        );
    }

    private ReplyContext replyContext(Message message) {
        if (!config.features.replyContext) {
            return ReplyContext.NONE;
        }
        Message referenced = message.getReferencedMessage();
        if (referenced == null) {
            return ReplyContext.NONE;
        }
        String quote = ReplyContextFormatter.quote(referenced.getContentDisplay());
        if (quote.isBlank() && !referenced.getAttachments().isEmpty()) {
            quote = "attachment: " + referenced.getAttachments().getFirst().getFileName();
        }
        if (quote.isBlank()) {
            quote = readableEmbedText(referenced.getEmbeds());
        }
        return new ReplyContext(authorName(referenced), ReplyContextFormatter.quote(quote));
    }

    private String readableEmbedText(List<MessageEmbed> embeds) {
        if (embeds == null || embeds.isEmpty()) {
            return "";
        }
        List<String> parts = new ArrayList<>();
        for (MessageEmbed embed : embeds) {
            if (embed.getTitle() != null && !embed.getTitle().isBlank()) {
                parts.add(embed.getTitle());
            }
            if (embed.getDescription() != null && !embed.getDescription().isBlank()) {
                parts.add(embed.getDescription());
            }
            for (MessageEmbed.Field field : embed.getFields()) {
                if (field.getName() != null && field.getValue() != null) {
                    parts.add(field.getName() + ": " + field.getValue());
                }
            }
        }
        return ReplyContextFormatter.quote(String.join(" | ", parts));
    }

    private String authorName(Message message) {
        Member member = message.getMember();
        if (member != null) {
            return member.getEffectiveName();
        }
        return message.getAuthor().getName();
    }

    private Integer roleColor(Member member) {
        if (member == null || member.getColorRaw() == 0) {
            return null;
        }
        return member.getColorRaw();
    }

    private record ReplyContext(String authorName, String quote) {
        private static final ReplyContext NONE = new ReplyContext(null, null);
    }
}

