package dev.penguinzdiscordintegration.mixin;

import dev.penguinzdiscordintegration.PenguinzDiscordIntegrationMod;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.damagesource.DamageSource;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ServerPlayer.class)
public class DeathBridgeMixin {
    @Inject(method = "die", at = @At("TAIL"))
    private void penguinzdiscordintegration$onDeath(DamageSource source, CallbackInfo ci) {
        if (PenguinzDiscordIntegrationMod.deathListener() != null) {
            PenguinzDiscordIntegrationMod.deathListener().onDeath((ServerPlayer) (Object) this, source);
        }
    }
}

