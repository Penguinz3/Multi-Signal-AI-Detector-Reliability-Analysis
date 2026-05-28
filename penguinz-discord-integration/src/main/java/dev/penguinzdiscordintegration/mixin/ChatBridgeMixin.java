package dev.penguinzdiscordintegration.mixin;

import dev.penguinzdiscordintegration.PenguinzDiscordIntegrationMod;
import net.minecraft.network.chat.ChatType;
import net.minecraft.network.chat.PlayerChatMessage;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.ServerGamePacketListenerImpl;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ServerGamePacketListenerImpl.class)
public class ChatBridgeMixin {
    @Inject(method = "broadcastChatMessage", at = @At("HEAD"))
    private void penguinzdiscordintegration$onChat(PlayerChatMessage message, ServerPlayer sender, ChatType.Bound bound, CallbackInfo ci) {
        if (PenguinzDiscordIntegrationMod.minecraftChatBridge() != null) {
            PenguinzDiscordIntegrationMod.minecraftChatBridge().onMinecraftChat(sender, message.decoratedContent());
        }
    }
}

