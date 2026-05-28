package dev.penguinzdiscordintegration.formatting;

public final class ReplyContextFormatter {
    private static final int MAX_QUOTE_LENGTH = 80;

    private ReplyContextFormatter() {
    }

    public static String quote(String text) {
        String cleaned = clean(text);
        if (cleaned.length() <= MAX_QUOTE_LENGTH) {
            return cleaned;
        }
        return cleaned.substring(0, MAX_QUOTE_LENGTH - 1) + "...";
    }

    public static String clean(String text) {
        if (text == null || text.isBlank()) {
            return "";
        }
        return text.replaceAll("\\s+", " ").trim();
    }
}

