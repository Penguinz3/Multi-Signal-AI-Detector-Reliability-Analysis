# Penguinz Discord Integration

Server-side Fabric 1.21.x Minecraft to Discord bridge inspired by DiscordIntegration, implemented independently.

## Build

```powershell
.\build\gradle-9.5.1-dist\gradle-9.5.1\bin\gradle.bat build --no-daemon
```

The server jar is written to:

```text
build/libs/penguinz-discord-integration-0.1.0.jar
```

## Setup

Put the jar in the server `mods` folder with Fabric API. On first start, the mod writes:

```text
config/penguinz-discord-integration.json
```

Set at least:

- `botToken`
- `guildId`
- `channels.chat`
- `channels.events`
- `channels.voiceMessages`
- `channels.statusTopic`
- `discordInviteUrl`

The Discord bot should have the message content intent enabled. Recommended permissions are Send Messages, Embed Links, Read Message History, Attach Files, Use Slash Commands, Manage Webhooks for webhook mode, and Manage Channels for topic updates.

## Features

- Minecraft chat to Discord in bot or webhook mode.
- Webhook chat can use player names and player head avatars.
- Discord chat to Minecraft with colored text, attachment links, and reply context.
- Smart alias pings from Minecraft to Discord with mention limits and cooldowns.
- Rich advancement and death embeds with player heads.
- Server start and stop messages.
- Configurable join and leave logs.
- `/mc status` and `/mc players` Discord slash commands.
- `/discord` in-game invite command.
- Channel topic updates for online/offline status, uptime, player count, and TPS.
- Optional VoiceMessages integration through the public `VoiceMessageReceivedCallback` API.

VoiceMessages forwarding sends the created Minecraft voice message to the configured Discord channel as an embed plus `.ogg` Opus audio upload. It intentionally does not include recipients, delete buttons, moderation buttons, coordinates, or live voice-channel bridge behavior.
