package dev.penguinzdiscordintegration.minecraft;

import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerPlayer;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Queue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicReference;

public final class ServerStatusProvider {
    private static final int MAX_TICK_SAMPLES = 100;

    private final AtomicReference<MinecraftServer> server = new AtomicReference<>();
    private final Queue<Long> tickIntervals = new ArrayDeque<>();
    private Instant startedAt;
    private long lastTickNanos;
    private int lastPlayersOnline;
    private int lastMaxPlayers;

    public void onServerStarted(MinecraftServer minecraftServer) {
        server.set(minecraftServer);
        startedAt = Instant.now();
        lastTickNanos = 0L;
        synchronized (tickIntervals) {
            tickIntervals.clear();
        }
    }

    public void onServerStopped() {
        server.set(null);
    }

    public void recordTick(MinecraftServer minecraftServer) {
        if (server.get() != minecraftServer) {
            return;
        }
        long now = System.nanoTime();
        if (lastTickNanos > 0L) {
            synchronized (tickIntervals) {
                tickIntervals.add(now - lastTickNanos);
                while (tickIntervals.size() > MAX_TICK_SAMPLES) {
                    tickIntervals.poll();
                }
            }
        }
        lastTickNanos = now;
    }

    public CompletableFuture<ServerSnapshot> snapshotAsync() {
        MinecraftServer minecraftServer = server.get();
        if (minecraftServer == null) {
            return CompletableFuture.completedFuture(ServerSnapshot.offline(lastPlayersOnline, lastMaxPlayers));
        }
        if (minecraftServer.isSameThread()) {
            return CompletableFuture.completedFuture(capture(minecraftServer));
        }

        CompletableFuture<ServerSnapshot> future = new CompletableFuture<>();
        minecraftServer.execute(() -> {
            try {
                future.complete(capture(minecraftServer));
            } catch (Exception e) {
                future.completeExceptionally(e);
            }
        });
        return future;
    }

    public void executeOnServerThread(ServerThreadTask task) {
        MinecraftServer minecraftServer = server.get();
        if (minecraftServer != null) {
            minecraftServer.execute(() -> task.run(minecraftServer));
        }
    }

    private ServerSnapshot capture(MinecraftServer minecraftServer) {
        List<String> players = new ArrayList<>();
        for (ServerPlayer player : minecraftServer.getPlayerList().getPlayers()) {
            players.add(player.getGameProfile().name());
        }

        int playersOnline = minecraftServer.getPlayerCount();
        int maxPlayers = minecraftServer.getMaxPlayers();
        lastPlayersOnline = playersOnline;
        lastMaxPlayers = maxPlayers;

        return new ServerSnapshot(
                true,
                playersOnline,
                maxPlayers,
                players,
                startedAt == null ? Duration.ZERO : Duration.between(startedAt, Instant.now()),
                currentTps()
        );
    }

    private double currentTps() {
        synchronized (tickIntervals) {
            if (tickIntervals.isEmpty()) {
                return 20.0D;
            }
            double averageNanos = tickIntervals.stream().mapToLong(Long::longValue).average().orElse(50_000_000D);
            if (averageNanos <= 0D) {
                return 20.0D;
            }
            return Math.min(20.0D, 1_000_000_000D / averageNanos);
        }
    }

    public static String formatDuration(Duration duration) {
        long seconds = Math.max(0L, duration.getSeconds());
        long days = seconds / 86_400L;
        seconds %= 86_400L;
        long hours = seconds / 3_600L;
        seconds %= 3_600L;
        long minutes = seconds / 60L;
        seconds %= 60L;

        if (days > 0L) {
            return String.format(Locale.US, "%dd %dh %dm", days, hours, minutes);
        }
        if (hours > 0L) {
            return String.format(Locale.US, "%dh %dm", hours, minutes);
        }
        if (minutes > 0L) {
            return String.format(Locale.US, "%dm %ds", minutes, seconds);
        }
        return seconds + "s";
    }

    @FunctionalInterface
    public interface ServerThreadTask {
        void run(MinecraftServer server);
    }

    public record ServerSnapshot(
            boolean online,
            int playersOnline,
            int maxPlayers,
            List<String> players,
            Duration uptime,
            double tps
    ) {
        public static ServerSnapshot offline(int lastPlayersOnline, int lastMaxPlayers) {
            return new ServerSnapshot(false, lastPlayersOnline, lastMaxPlayers, List.of(), Duration.ZERO, 0D);
        }
    }
}

