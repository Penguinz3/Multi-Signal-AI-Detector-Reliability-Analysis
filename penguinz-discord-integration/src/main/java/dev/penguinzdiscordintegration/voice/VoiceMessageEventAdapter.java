package dev.penguinzdiscordintegration.voice;

import dev.penguinzdiscordintegration.PenguinzDiscordIntegrationMod;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.List;

public final class VoiceMessageEventAdapter {
    private static final String CALLBACK_CLASS = "ru.dimaskama.voicemessages.api.VoiceMessageReceivedCallback";

    public boolean register(VoiceMessageConsumer consumer) {
        try {
            Class<?> callbackClass = Class.forName(CALLBACK_CLASS);
            Object event = callbackClass.getField("EVENT").get(null);
            Method register = event.getClass().getMethod("register", Object.class);
            Object proxy = Proxy.newProxyInstance(
                    callbackClass.getClassLoader(),
                    new Class<?>[]{callbackClass},
                    handler(consumer)
            );
            register.invoke(event, proxy);
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        } catch (ReflectiveOperationException | LinkageError e) {
            PenguinzDiscordIntegrationMod.LOGGER.warn("VoiceMessages API was present but could not be registered", e);
            return false;
        }
    }

    private InvocationHandler handler(VoiceMessageConsumer consumer) {
        return (proxy, method, args) -> {
            if ("toString".equals(method.getName())) {
                return "PenguinzDiscordIntegrationVoiceMessagesCallback";
            }
            if ("hashCode".equals(method.getName())) {
                return System.identityHashCode(proxy);
            }
            if ("equals".equals(method.getName())) {
                return proxy == args[0];
            }
            if ("onVoiceMessageReceived".equals(method.getName()) && args != null && args.length == 3) {
                @SuppressWarnings("unchecked")
                List<byte[]> frames = (List<byte[]>) args[1];
                consumer.onVoiceMessage(args[0], frames, (String) args[2]);
                return false;
            }
            return method.getDefaultValue();
        };
    }

    @FunctionalInterface
    public interface VoiceMessageConsumer {
        void onVoiceMessage(Object sender, List<byte[]> opusFrames, String targetName);
    }
}

