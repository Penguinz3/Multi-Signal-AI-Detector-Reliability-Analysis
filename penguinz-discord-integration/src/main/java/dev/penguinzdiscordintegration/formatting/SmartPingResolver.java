package dev.penguinzdiscordintegration.formatting;

import dev.penguinzdiscordintegration.config.BridgeConfig;

import java.time.Duration;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class SmartPingResolver {
    private static final Pattern ALIAS = Pattern.compile("(?<![\\w])@([A-Za-z0-9_.-]{2,32})\\b");
    private static final Pattern EVERYONE_HERE = Pattern.compile("(?i)@(everyone|here)\\b");
    private static final Pattern ROLE_MENTION = Pattern.compile("<@&(\\d{5,25})>");

    private final BridgeConfig config;
    private final ConcurrentMap<UUID, Instant> lastMentionByPlayer = new ConcurrentHashMap<>();

    public SmartPingResolver(BridgeConfig config) {
        this.config = config;
    }

    public Result resolve(UUID minecraftPlayer, String message) {
        String safeMessage = sanitizeDisallowedMentions(message == null ? "" : message);
        if (!config.features.smartPings || !config.smartPings.enabled || config.smartPings.aliases.isEmpty()) {
            return new Result(safeMessage, Set.of());
        }

        Instant now = Instant.now();
        if (isCoolingDown(minecraftPlayer, now)) {
            return new Result(safeMessage, Set.of());
        }

        Map<String, String> aliases = normalizedAliases();
        int maxMentions = Math.max(0, config.smartPings.maxMentionsPerMessage);
        if (maxMentions == 0) {
            return new Result(safeMessage, Set.of());
        }

        Matcher matcher = ALIAS.matcher(safeMessage);
        StringBuilder resolved = new StringBuilder();
        Set<String> allowedUserMentions = new LinkedHashSet<>();
        int mentionCount = 0;

        while (matcher.find()) {
            String alias = matcher.group(1).toLowerCase(Locale.ROOT);
            String discordUserId = aliases.get(alias);
            if (discordUserId == null || mentionCount >= maxMentions) {
                matcher.appendReplacement(resolved, Matcher.quoteReplacement(matcher.group(0)));
                continue;
            }

            mentionCount++;
            allowedUserMentions.add(discordUserId);
            matcher.appendReplacement(resolved, Matcher.quoteReplacement("<@" + discordUserId + ">"));
        }
        matcher.appendTail(resolved);

        if (!allowedUserMentions.isEmpty() && config.smartPings.cooldownSeconds > 0) {
            lastMentionByPlayer.put(minecraftPlayer, now);
        }

        return new Result(resolved.toString(), allowedUserMentions);
    }

    private boolean isCoolingDown(UUID player, Instant now) {
        int cooldownSeconds = config.smartPings.cooldownSeconds;
        if (cooldownSeconds <= 0) {
            return false;
        }
        Instant lastMention = lastMentionByPlayer.get(player);
        return lastMention != null && Duration.between(lastMention, now).getSeconds() < cooldownSeconds;
    }

    private Map<String, String> normalizedAliases() {
        Map<String, String> aliases = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : config.smartPings.aliases.entrySet()) {
            String alias = entry.getKey();
            String id = entry.getValue();
            if (alias == null || id == null || alias.isBlank() || !id.matches("\\d{5,25}")) {
                continue;
            }
            aliases.put(alias.toLowerCase(Locale.ROOT), id);
        }
        return aliases;
    }

    private String sanitizeDisallowedMentions(String message) {
        String sanitized = message;
        if (!config.smartPings.allowEveryoneHere) {
            sanitized = EVERYONE_HERE.matcher(sanitized).replaceAll(match -> "@\u200B" + match.group(1));
        }
        if (!config.smartPings.allowRolePings) {
            sanitized = ROLE_MENTION.matcher(sanitized).replaceAll(match -> "<@\u200B&" + match.group(1) + ">");
        }
        return sanitized;
    }

    public record Result(String content, Set<String> allowedUserMentions) {
    }
}

