package dev.penguinzdiscordintegration.mixin;

import dev.penguinzdiscordintegration.PenguinzDiscordIntegrationMod;
import net.minecraft.advancements.AdvancementHolder;
import net.minecraft.server.PlayerAdvancements;
import net.minecraft.server.level.ServerPlayer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(PlayerAdvancements.class)
public class AdvancementBridgeMixin {
    @Shadow
    private ServerPlayer player;

    @Inject(
            method = "award",
            at = @At(value = "INVOKE", target = "Lnet/minecraft/server/PlayerAdvancements;markForVisibilityUpdate(Lnet/minecraft/advancements/AdvancementHolder;)V")
    )
    private void penguinzdiscordintegration$onAdvancement(AdvancementHolder advancementHolder, String criterionName, CallbackInfoReturnable<Boolean> cir) {
        if (PenguinzDiscordIntegrationMod.advancementListener() != null) {
            PenguinzDiscordIntegrationMod.advancementListener().onAdvancement(player, advancementHolder);
        }
    }
}

