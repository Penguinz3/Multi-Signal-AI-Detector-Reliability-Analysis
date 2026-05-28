package dev.penguinzdiscordintegration.discord;

import dev.penguinzdiscordintegration.PenguinzDiscordIntegrationMod;
import dev.penguinzdiscordintegration.config.BridgeConfig;
import net.dv8tion.jda.api.entities.Message;
import net.dv8tion.jda.api.entities.Webhook;
import net.dv8tion.jda.api.entities.channel.concrete.TextChannel;
import net.dv8tion.jda.api.requests.restaction.WebhookMessageCreateAction;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;

public final class WebhookManager {
    private static final String WEBHOOK_NAME = "Penguinz Discord Integration";

    private final BridgeConfig config;
    private volatile Webhook cachedWebhook;
    private volatile CompletableFuture<Webhook> webhookLookup;

    public WebhookManager(BridgeConfig config) {
        this.config = config;
    }

    public void send(
            TextChannel channel,
            String username,
            String avatarUrl,
            String content,
            Collection<String> allowedUserMentions,
            Runnable fallback
    ) {
        getOrCreateWebhook(channel).whenComplete((webhook, error) -> {
            if (error != null || webhook == null) {
                PenguinzDiscordIntegrationMod.LOGGER.warn("Webhook unavailable for channel {}; falling back to bot message.", channel.getId(), error);
                fallback.run();
                return;
            }

            try {
                WebhookMessageCreateAction<Message> action = webhook.sendMessage(content)
                        .setAllowedMentions(Collections.emptySet());
                if (username != null && !username.isBlank()) {
                    action.setUsername(username.length() > 80 ? username.substring(0, 80) : username);
                }
                if (avatarUrl != null && !avatarUrl.isBlank()) {
                    action.setAvatarUrl(avatarUrl);
                }
                if (allowedUserMentions != null && !allowedUserMentions.isEmpty()) {
                    action.mentionUsers(allowedUserMentions);
                }
                action.queue(
                        message -> { },
                        sendError -> {
                            PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to send webhook message; falling back to bot message.", sendError);
                            fallback.run();
                        }
                );
            } catch (Exception e) {
                PenguinzDiscordIntegrationMod.LOGGER.warn("Failed to prepare webhook message; falling back to bot message.", e);
                fallback.run();
            }
        });
    }

    private CompletableFuture<Webhook> getOrCreateWebhook(TextChannel channel) {
        Webhook existing = cachedWebhook;
        if (existing != null && existing.getChannel() != null && channel.getId().equals(existing.getChannel().getId())) {
            return CompletableFuture.completedFuture(existing);
        }

        synchronized (this) {
            if (webhookLookup != null && !webhookLookup.isDone()) {
                return webhookLookup;
            }
            webhookLookup = new CompletableFuture<>();
        }

        channel.retrieveWebhooks().queue(
                webhooks -> useExistingOrCreate(channel, webhooks, webhookLookup),
                error -> {
                    PenguinzDiscordIntegrationMod.LOGGER.warn("Could not fetch Discord webhooks for channel {}. Missing Manage Webhooks?", channel.getId(), error);
                    webhookLookup.complete(null);
                }
        );
        return webhookLookup;
    }

    private void useExistingOrCreate(TextChannel channel, List<Webhook> webhooks, CompletableFuture<Webhook> future) {
        for (Webhook webhook : webhooks) {
            if (WEBHOOK_NAME.equals(webhook.getName())) {
                cachedWebhook = webhook;
                future.complete(webhook);
                return;
            }
        }

        channel.createWebhook(WEBHOOK_NAME).queue(
                webhook -> {
                    cachedWebhook = webhook;
                    future.complete(webhook);
                },
                error -> {
                    PenguinzDiscordIntegrationMod.LOGGER.warn("Could not create Discord webhook for channel {}. Missing Manage Webhooks?", channel.getId(), error);
                    future.complete(null);
                }
        );
    }
}

